"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict
import numpy as np
torch.cuda.empty_cache()


from ops import compute_spatial_ratio_encodings, binary_focal_loss
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger()  

import sys
sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs/heads/TransH')
from TransH import TransH
from NegativeSampling import NegativeSampling
from MarginLoss import MarginLoss

class InteractionHead(Module):
    """Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_roi_pool: Module
        Module that performs RoI pooling or its variants
    box_pair_head: Module
        Module that constructs and computes box pair features
    box_pair_suppressor: Module
        Module that computes unary weights for each box pair
    box_pair_predictor: Module
        Module that classifies box pairs
    human_idx: int
        The index of human/person class in all objects
    num_classes: int
        Number of target classes
    box_nms_thresh: float, default: 0.5
        Threshold used for non-maximum suppression
    box_score_thresh: float, default: 0.2
        Threshold used to filter out low-quality boxes
    max_human: int, default: 15
        Number of human detections to keep in each image
    max_object: int, default: 15
        Number of object (excluding human) detections to keep in each image
    distributed: bool, default: False
        Whether the model is trained under distributed data parallel. If True,
        the number of positive logits will be averaged across all subprocesses
    """
    def __init__(self,
        # Network components
        box_roi_pool: Module,
        box_pair_head: Module,
        box_pair_suppressor: Module,
        box_pair_predictor: Module,
        # Dataset properties
        human_idx: int,
        num_classes: int,
        # Hyperparameters
        box_nms_thresh: float = 0.5,
        box_score_thresh: float = 0.2,
        max_human: int = 15,
        max_object: int = 15,
        # Misc
        distributed: bool = False
    ) -> None:
        super().__init__()

        self.box_roi_pool = box_roi_pool
        self.box_pair_head = box_pair_head
        self.box_pair_suppressor = box_pair_suppressor
        self.box_pair_predictor = box_pair_predictor

        self.num_classes = num_classes
        self.human_idx = human_idx

        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.max_human = max_human
        self.max_object = max_object

        self.distributed = distributed

    def preprocess(self,
        detections: List[dict],
        targets: List[dict],
        append_gt: Optional[bool] = None
    ) -> None:

        results = []
        for b_idx, detection in enumerate(detections):
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']

            # Append ground truth during training
            if append_gt is None:
                append_gt = self.training
            if append_gt:
                target = targets[b_idx]
                n = target["boxes_h"].shape[0]
                boxes = torch.cat([target["boxes_h"], target["boxes_o"], boxes])
                scores = torch.cat([torch.ones(2 * n, device=scores.device), scores])
                labels = torch.cat([
                    self.human_idx * torch.ones(n, device=labels.device).long(),
                    target["object"],
                    labels
                ])

            # Remove low scoring examples
            active_idx = torch.nonzero(
                scores >= self.box_score_thresh
            ).squeeze(1)
            # Class-wise non-maximum suppression
            keep_idx = box_ops.batched_nms(
                boxes[active_idx],
                scores[active_idx],
                labels[active_idx],
                self.box_nms_thresh
            )
            active_idx = active_idx[keep_idx]
            # Sort detections by scores
            sorted_idx = torch.argsort(scores[active_idx], descending=True)
            active_idx = active_idx[sorted_idx]
            # Keep a fixed number of detections
            h_idx = torch.nonzero(labels[active_idx] == self.human_idx).squeeze(1)
            o_idx = torch.nonzero(labels[active_idx] != self.human_idx).squeeze(1)
            if len(h_idx) > self.max_human:
                h_idx = h_idx[:self.max_human]
            if len(o_idx) > self.max_object:
                o_idx = o_idx[:self.max_object]
            # Permute humans to the top
            keep_idx = torch.cat([h_idx, o_idx])
            active_idx = active_idx[keep_idx]

            results.append(dict(
                boxes=boxes[active_idx].view(-1, 4),
                labels=labels[active_idx].view(-1),
                scores=scores[active_idx].view(-1)
            
            ))

        return results

    def compute_interaction_classification_loss(self, results: List[dict]) -> Tensor:
        scores = []; labels = []
        for result in results:
            score = result['scores']
            label = result['labels']
            # logger.info(f'scores,labels: {len(score)},{len(label)}')
            scores.append(score)
            labels.append(label)
            
        labels = torch.cat(labels)
        scores = torch.cat(scores)

       
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            scores, labels, reduction='sum', gamma=0.2
        )/n_p
        # logger.info(f'compute_classification_loss:{loss}')
        return loss 


    def compute_interactiveness_loss(self, results:List[dict]) -> Tensor:
        weights = []; labels = []
        for result in results:
            weight = result['weights']
            label = result['unary_labels']
            # logger.info(f'weights,labels: {len(weights)},{len(labels)}')
            weights.append(weight)
            labels.append(label)
            

        weights = torch.cat(weights)
        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            weights, labels, reduction='sum', gamma=2.0
        ) / n_p
        # logger.info(f'compute_interactiveness_loss:{loss}')
        # logger.info(f'loss:{loss}')
        return loss

    def compute_transH_loss(self, 
        positive_scores:List[Tensor],
        negative_scores:List[Tensor],
        head:List[Tensor],
        relation:List[Tensor],
        relation_norm:List[Tensor],
        tail:List[Tensor],
        results: List[dict]
    ) -> Tensor:

        labels = []
        for result in results:
            label = result['unary_labels']
            labels.append(label)
        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        model = NegativeSampling(
            loss = MarginLoss(margin = 4.0),
            batch_size = 256
        ).to('cuda')
        loss = model(positive_scores,negative_scores,head,relation,relation_norm,tail)/n_p
        # logger.info(f'transH_loss:{loss}')
        return loss

    def postprocess(self,
        logits_p: Tensor,
        logits_s: Tensor,
        prior: List[Tensor],
        boxes_h: List[Tensor],
        boxes_o: List[Tensor],
        object_class: List[Tensor],
        labels: List[Tensor]
    ) -> Tuple[
        List[dict]
        ]:
        # ) -> List[dict]:
        """
        Parameters:
        -----------
        logits_p: Tensor
            (N, K) Classification logits on each action for all box pairs
        logits_s: Tensor
            (N, 1) Logits for unary weights
        prior: List[Tensor]
            Prior scores organised by images. Each tensor has shape (2, M, K).
            M could be different for different images
        boxes_h: List[Tensor]
            Human bounding box coordinates organised by images (M, 4)
        boxes_o: List[Tensor]
            Object bounding box coordinates organised by images (M, 4)
        object_classes: List[Tensor]
            Object indices for each pair organised by images (M,)
        labels: List[Tensor]
            Binary labels on each action organised by images (M, K)

        Returns:
        --------
        results: List[dict]
            Results organised by images, with keys as below
            `boxes_h`: Tensor[M, 4]
            `boxes_o`: Tensor[M, 4]
            `index`: Tensor[L]
                Expanded indices of box pairs for each predicted action
            `prediction`: Tensor[L]
                Expanded indices of predicted actions
            `scores`: Tensor[L]
                Scores for each predicted action
            `object`: Tensor[M]
                Object indices for each pair
            `prior`: Tensor[2, L]
                Prior scores for expanded pairs
            `weights`: Tensor[M]
                Unary weights for each box pair
            `labels`: Tensor[L], optional
                Binary labels on each action
            `unary_labels`: Tensor[M], optional
                Labels for the unary weights
        """
        num_boxes = [len(b) for b in boxes_h]
        # logger.info(f'transh_score:{transh_score}')
        weights = torch.sigmoid(logits_s).squeeze(1)
        # logger.info(f'min of x: {min(x)}')
        scores = torch.sigmoid(logits_p)
        weights = weights.split(num_boxes)
        scores = scores.split(num_boxes)
        if len(labels) == 0:
            labels = [None for _ in range(len(num_boxes))]
        # if len(transH_positive_scores) == 0:
        #     transH_positive_scores = [None for _ in range(len(num_boxes))]
        # if len(transH_negative_scores) == 0:
        #     transH_negative_scores = [None for _ in range(len(num_boxes))]

        results = []
        # transH_positive_score_list = []
        # transH_negative_score_list = []
        for w, s, p, b_h, b_o, o, l in zip(
            weights, scores, prior, boxes_h, boxes_o, object_class, labels
        ):
        # for w, s, p, b_h, b_o, o, l in zip(
        #     weights, scores, prior, boxes_h, boxes_o, object_class, labels
        #     ):
            # Keep valid classes
            x, y = torch.nonzero(p[0]).unbind(1)
            action_score = s[x, y] * p[:, x, y].prod(dim=0) * w[x].detach()
            result_dict = dict(
                boxes_h=b_h, boxes_o=b_o,
                index=x, prediction=y,
                scores=action_score,
                object=o, prior=p[:, x, y], weights=w
            )
            # logger.info(f'results:{result_dict}')
            # If binary labels are provided
            if l is not None:
                ##one to many verb prediction
                action_binary_label = l[x, y] 
                result_dict['labels'] = action_binary_label 
                ##whether pair is valid
                result_dict['unary_labels'] = l.sum(dim=1).clamp(max=1)
 

            results.append(result_dict)
            # transH_positive_score_list.append(t_p)
            # transH_negative_score_list.append(t_n)

        return results
        # return results


    def forward(self,
        features: OrderedDict,
        detections: List[dict],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        detections: List[dict]
            Object detections with the following keys
            `boxes`: Tensor[N, 4]
            `labels`: Tensor[N]
            `scores`: Tensor[N]
        image_shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        targets: List[dict], optional
            Interaction targets with the following keys
            `boxes_h`: Tensor[G, 4]
            `boxes_o`: Tensor[G, 4]
            `object`: Tensor[G]
                Object class indices for each pair
            `labels`: Tensor[G]
                Target class indices for each pair

        Returns:
        --------
        results: List[dict]
            Results organised by images. During training the loss dict is appended to the
            end of the list, resulting in the length being larger than the number of images
            by one. For the result dict of each image, refer to `postprocess` for documentation.
            The loss dict has two keys
            `hoi_loss`: Tensor
                Loss for HOI classification
            `interactiveness_loss`: Tensor
                Loss incurred on learned unary weights
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        detections = self.preprocess(detections, targets)

        box_coords = [detection['boxes'] for detection in detections]
        box_labels = [detection['labels'] for detection in detections]
        box_scores = [detection['scores'] for detection in detections]
        box_features = self.box_roi_pool(features, box_coords, image_shapes)
        # logger.info(f'box_features:{box_features}')
        if self.training:

            box_pair_features, boxes_h, boxes_o, object_class,\
            box_pair_labels, box_pair_prior,transH_positive_scores,\
            transH_negative_scores, head_entitys,tail_entitys,relations,relations_norm = self.box_pair_head(
                features, image_shapes, box_features,
                box_coords, box_labels, box_scores, targets
            )
        else:
            box_pair_features, boxes_h, boxes_o, object_class,\
            box_pair_labels, box_pair_prior,transH_positive_scores,\
            transH_negative_scores = self.box_pair_head(
                features, image_shapes, box_features,
                box_coords, box_labels, box_scores, targets)
        # box_pair_features, boxes_h, boxes_o, object_class, box_pair_labels, box_pair_prior = self.box_pair_head(
        #     features, image_shapes, box_features,
        #     box_coords, box_labels, box_scores, targets
        # )
        # logger.info(f'box_pair_features:{box_pair_features[0].size()},{box_pair_features[1].size()},{box_pair_features[2].size()},{box_pair_features[3].size()}')
        box_pair_features = torch.cat(box_pair_features)

        logits_p = self.box_pair_predictor(box_pair_features)
        logits_s = self.box_pair_suppressor(box_pair_features)
        ##logits_p,logits_s: predictors
        # boxes_h,boxes_o, box_pair_prior, box_pair_labels,object_class: targets
        results = self.postprocess(
            logits_p, logits_s, box_pair_prior,
            boxes_h, boxes_o,
            object_class, box_pair_labels
        )
        if self.training:
            loss_dict = dict(
                hoi_loss=self.compute_interaction_classification_loss(results),
                interactiveness_loss=self.compute_interactiveness_loss(results),
                transH_loss = self.compute_transH_loss(transH_positive_scores,transH_negative_scores,\
                                                       head_entitys,relations,relations_norm,tail_entitys,results)
            )
            # logger.info(f'compute interactiveness loss:{loss_dict.values()}')
            results.append(loss_dict)

        return results

class MultiBranchFusion(Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        representation_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)) #(1,14,1024)

class MessageMBF(MultiBranchFusion):
    """
    MBF for the computation of anisotropic messages

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    node_type: str
        Nature of the sending node. Choose between `human` amd `object`
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int,
        spatial_size: int,
        representation_size: int,
        node_type: str,
        cardinality: int
    ) -> None:
        super().__init__(appearance_size, spatial_size, representation_size, cardinality)

        if node_type == 'human':
            self._forward_method = self._forward_human_nodes
        elif node_type == 'object':
            self._forward_method = self._forward_object_nodes
        else:
            raise ValueError("Unknown node type \"{}\"".format(node_type))

    def _forward_human_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        assert len(appearance) == n_h, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n, 1, 1)
                * fc_2(spatial).permute([1, 0, 2])
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)
    def _forward_object_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_h, n = spatial.shape[:2]
        # logger.info(f'size of dim0 for appearance features:{len(appearance)}')
        assert len(appearance) == n, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n_h, 1, 1)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def forward(self, *args) -> Tensor:
        return self._forward_method(*args)


class transH_head(Module):

    def __init__(self,
        transh_dim: int = 100,
        transh_p_norm: int = 2,
        transh_norm_flag: bool = True,
        human_idx: int = 1, 
        num_object: int = 80,
        num_cls: int = 24
    ) -> None:

        super().__init__()
        self.transh_dim = transh_dim
        self.transh_p_norm = transh_p_norm
        self.transh_norm_flag = transh_norm_flag
        self.device = 'cuda'
        self.human_idx = human_idx
        self.num_object = num_object
        self.num_cls = num_cls
        # self.fc_relation = nn.Sequential(
        #     nn.Linear(37, 1024),
        #     nn.ReLU(),
        # )


    def forward(self,
        ind_x: list,
        ind_y: list,
        n_h: int,
        n:int,
        x_keep: list,
        y_keep: list,
        labels: Optional[Tensor] = None
        ) -> Tuple[
        Tensor, Tensor,Tensor, Tensor,Module
        ]:

        relations = torch.tensor([*range(0,self.num_cls)],device=self.device, dtype=torch.int64).repeat(len(ind_x))
        # logger.info(f'relations:{relations.size()}')
        heads = torch.tensor([self.human_idx],device=self.device, dtype=torch.int64).repeat(len(ind_x)*self.num_cls)
        # logger.info(f'heads:{heads.size()}')
        tails = torch.tensor(ind_y.repeat_interleave(self.num_cls),device=self.device, dtype=torch.int64) #.repeat(len(ind_x))
        # logger.info(f'tails:{tails}')
        transH= TransH(ent_tot = self.num_object,
                        rel_tot = self.num_cls, 
                        dim = self.transh_dim, 
                        p_norm = self.transh_p_norm, 
                        norm_flag = self.transh_norm_flag).to(self.device)
            
        head_entity,relation,relation_norm,tail_entity,transh_scores = transH(heads,relations,tails)
        # logger.info(f'head_entity,tail_entity,transh_scores:{head_entity.size()},{tail_entity.size()},{transh_scores.size()}')
       
        # logger.info(f'transh_tail_entity 0 and -1:{transh_tail_entitys[0]},{transh_tail_entitys[-1]}')
        # transh_head_entitys_flatten = torch.flatten(transh_head_entitys_stack, start_dim=1,end_dim=-1)
        # transh_tail_entitys_flatten = torch.flatten(transh_tail_entitys_stack, start_dim=1,end_dim=-1)
        # transh_head_entitys_fc = self.fc_head(transh_head_entitys_flatten)
        # transh_tail_entitys_fc  = self.fc_tail(transh_tail_entitys_flatten)
        # logger.info(f'transh_socres:{transh_scores}')

        return head_entity,relation,relation_norm,tail_entity,transh_scores



class GraphHead(Module):
    """
    Graphical model head

    Parameters:
    -----------
    output_channels: int
        Number of output channels of the backbone
    roi_pool_size: int
        Spatial resolution of the pooled output
    node_encoding_size: int
        Size of the node embeddings
    num_cls: int
        Number of targe classes
    human_idx: int
        The index of human/person class in all objects
    object_class_to_target_class: List[list]
        The mapping (potentially one-to-many) from objects to target classes
    fg_iou_thresh: float, default: 0.5
        The IoU threshold to identify a positive example
    num_iter: int, default 2
        Number of iterations of the message passing process
    """
    def __init__(self,
        out_channels: int,
        roi_pool_size: int,
        node_encoding_size: int, 
        representation_size: int, 
        num_cls: int, human_idx: int,
        object_class_to_target_class: List[list],
        fg_iou_thresh: float = 0.5,
        num_iter: int = 2
    ) -> None:

        super().__init__()

        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Box head to map RoI features to low dimensional
        self.box_head = nn.Sequential(
            Flatten(start_dim=1),
            nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
            nn.ReLU(),
            nn.Linear(node_encoding_size, node_encoding_size),
            nn.ReLU()
        )

        # Compute adjacency matrix
        self.adjacency = nn.Linear(representation_size, 1)

        # Compute messages
        self.sub_to_obj = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='human',
            cardinality=16
        )
        self.obj_to_sub = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='object',
            cardinality=16
        )

        self.norm_h = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(46, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        # Spatial attention head
        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Attention head for global features
        self.attention_head_g = MultiBranchFusion(
            256, 1024,
            representation_size, cardinality=16
        )

        self.transh_head = transH_head(
            transh_dim = 70,
            transh_p_norm = 2,
            transh_norm_flag = True,
            human_idx = self.human_idx,
            num_object = 80,
            num_cls = self.num_cls
            )

        self.fc_head = nn.Sequential(
            nn.Linear(1094, 1024),
            nn.ReLU(),
        )
        self.fc_tail = nn.Sequential(
            nn.Linear(1094, 1024),
            nn.ReLU(),
        )

    def associate_with_ground_truth(self,
        boxes_h: Tensor,
        boxes_o: Tensor,
        targets: List[dict]
    ) -> Tensor:
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_h.device)

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)
        # logger.info(f'x,y:{x},{y}')
        labels[x, targets["labels"][y]] = 1
        verb_label = targets["labels"][y]
        
        return labels

    def compute_prior_scores(self,
        x: Tensor, y: Tensor,
        scores: Tensor,
        object_class: Tensor
    ) -> Tensor:
        """
        Parameters:
        -----------
            x: Tensor[M]
                Indices of human boxes (paired)
            y: Tensor[M]
                Indices of object boxes (paired)
            scores: Tensor[N]
                Object detection scores (before pairing)
            object_class: Tensor[N]
                Object class indices (before pairing)
        """
        prior_h = torch.zeros(len(x), self.num_cls, device=scores.device) #(N,117)
        prior_o = torch.zeros_like(prior_h) #(N,117)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p) #(N,1)
        s_o = scores[y].pow(p) #(N,1)
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        # logger.info(f'object_class[y]:{object_class[y]}')
        # logger.info(f'self.object_class_to_target_class:{self.object_class_to_target_class}')
        target_cls_idx = [self.object_class_to_target_class[obj.item()] for obj in object_class[y]]
        # logger.info(f'self.object_class_to_target_class:{self.object_class_to_target_class}')
        

        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # logger.info(f'pair_idx:{pair_idx}')
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]
        # logger.info(f'flat_target_idx:{flat_target_idx}')

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]
        # logger.info(f'prior_h:{prior_h.size()}')
        # logger.info(f'prior_o:{prior_o.size()}')

        final_score = torch.stack([prior_h, prior_o])
        # logger.info(f'final_score:{final_score.size()}')

        return final_score

    def forward(self,
        features: OrderedDict, image_shapes: List[Tuple[int, int]],
        box_features: Tensor, box_coords: List[Tensor],
        box_labels: List[Tensor], box_scores: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor], List[Tensor]
    ]:
        """
        Parameters:
        -----------
            features: OrderedDict
                Feature maps returned by FPN
            box_features: Tensor
                (N, C, P, P) Pooled box features
            image_shapes: List[Tuple[int, int]]
                Image shapes, heights followed by widths
            box_coords: List[Tensor]
                Bounding box coordinates organised by images
            box_labels: List[Tensor]
                Bounding box object types organised by images
            box_scores: List[Tensor]
                Bounding box scores organised by images
            targets: List[dict]
                Interaction targets with the following keys
                `boxes_h`: Tensor[G, 4]
                `boxes_o`: Tensor[G, 4]
                `labels`: Tensor[G]

        Returns:
        --------
            all_box_pair_features: List[Tensor]
            all_boxes_h: List[Tensor]
            all_boxes_o: List[Tensor]
            all_object_class: List[Tensor]
            all_labels: List[Tensor]
            all_prior: List[Tensor]
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"
        ###batch images
        global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        box_features = self.box_head(box_features)

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        all_boxes_h = []; all_boxes_o = []; all_object_class = []
        all_labels = []; all_prior = []
        all_box_pair_features = []; all_positive_transH_scores = []; all_negative_transH_scores = []
        all_head_entitys = []; all_tail_entitys = []; all_relations = [];all_relations_norm = []
        ### singer image
        for b_idx, (coords, labels, scores, bbox_features) in enumerate(zip(box_coords, box_labels, box_scores,box_features)):
            n = num_boxes[b_idx] #3
            # logger.info(f'box_features:{bbox_features}')
            device = box_features.device
            n_h = torch.sum(labels == self.human_idx).item()
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                all_box_pair_features.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                all_boxes_h.append(torch.zeros(0, 4, device=device))
                all_boxes_o.append(torch.zeros(0, 4, device=device))
                all_object_class.append(torch.zeros(0, device=device, dtype=torch.int64))
                all_prior.append(torch.zeros(2, 0, self.num_cls, device=device))
                all_labels.append(torch.zeros(0, self.num_cls, device=device))
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise ValueError("Human detections are not permuted to the top")

            node_encodings = box_features[counter: counter+n]
            # Duplicate human nodes
            h_node_encodings = node_encodings[:n_h] 
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
            # logger.info(f'x,y:{x},{y}')
            # Remove pairs consisting of the same human instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            # logger.info(f'x_keep, y_keep:{x_keep},{y_keep}')
            
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()
            # logger.info(f'x, y:{x},{y}')
            # Compute spatial features
            box_pair_spatial = compute_spatial_ratio_encodings(
                [coords[x]], [coords[y]], [image_shapes[b_idx]]
            )
            if torch.isnan(box_pair_spatial).sum() > 0:
                logger.info(f'produced the box_parir_spatial nan values: {box_pair_spatial}')
                box_pair_spatial = torch.nan_to_num(box_pair_spatial)
            ##graph embedding input

            transh_head_entitys,transh_relations,transh_relations_norm,transh_tail_entitys,transh_scores = self.transh_head(
                                                        x,y,n_h,n,x_keep,y_keep,labels
                                                        )
            # logger.info(f'transH:{transH}')
            head_entitys,tail_entitys = [],[]

            for idx,(entity_x,entity_y) in enumerate(zip(transh_head_entitys,transh_tail_entitys)):   
                # transh_scores.append(transh_sore)
                if idx % self.num_cls == 0:
                    head_entitys.append(entity_x)
                    tail_entitys.append(entity_y)
            transh_head_entitys_stack = torch.stack(head_entitys)
            transh_tail_entitys_stack = torch.stack(tail_entitys)
            graph_human_node = self.fc_head(torch.cat((h_node_encodings[x],transh_head_entitys_stack),1))
            graph_object_node = self.fc_tail(torch.cat((node_encodings[y],transh_tail_entitys_stack),1))
            # logger.info(f'graph_human_node,graph_object_node:{graph_human_node.size()},{graph_object_node.size()}')
            # Reshape the spatial features
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n_h, n, -1)

            adjacency_matrix = torch.ones(n_h, n, device=device)
            for _ in range(self.num_iter):
                # Compute weights of each edge
                app = torch.cat([graph_human_node,graph_object_node], 1)
                # logger.info(f'app:{app.size()}')
                weights = self.attention_head(app,box_pair_spatial)
                adjacency_matrix = self.adjacency(weights).reshape(n_h, n)
            
                # Update human nodes
                index_o = int((graph_object_node.size()[0]/n_h))
                index_h = [i for i in range(0,graph_human_node.size()[0]) if i % n == 0]
                # logger.info(f'indics_h :{index_h}')
                all_entity_encoding = graph_object_node[0:index_o]
                h_entity_encoding = graph_human_node[index_h]
                # logger.info(f'all_entity_encoding:{all_entity_encoding.size()}')
                # logger.info(f'h_entity_encoding:{h_entity_encoding.size()}')
                o_t_s = self.obj_to_sub(all_entity_encoding,box_pair_spatial_reshaped) 
                # logger.info(f'o_t_s:{o_t_s.size()}')
                messages_to_h = F.relu(torch.sum(
                    adjacency_matrix.softmax(dim=1)[..., None] * o_t_s, dim=1))

                h_node_encodings = self.norm_h(
                    h_entity_encoding + messages_to_h
                )
                # Update object nodes (including human nodes)
                messages_to_o = F.relu(torch.sum(
                    adjacency_matrix.t().softmax(dim=1)[..., None] *
                    self.sub_to_obj(
                        h_entity_encoding,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                node_encodings = self.norm_o(
                    all_entity_encoding + messages_to_o
                )
            transh_scores_keep= transh_scores.reshape(n_h,n,self.num_cls)[x_keep,y_keep]
            transh_head_entitys_keep= transh_head_entitys.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            transh_tail_entitys_keep= transh_tail_entitys.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            transh_relations_keep = transh_relations.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            transh_relations_norm_keep = transh_relations_norm.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            # logger.info(f'transH_scores_keep:{transh_scores_keep.size()}')

            if targets is not None:
                target_label = self.associate_with_ground_truth(coords[x_keep], coords[y_keep], targets[b_idx])
                # logger.info(f'target_labels:{target_label.size()}')
                positive_x, positive_y = torch.nonzero(target_label).unbind(1)
                # logger.info(f'positive_x, positive_y:{positive_x}, {positive_y}')
                negative_xy = (target_label== 0).nonzero()
                rand_columns = torch.randperm(negative_xy.size()[0])[:len(positive_x)]
                # logger.info(f'rand_columns:{rand_columns}')
                negative_x,negative_y = negative_xy[rand_columns].unbind(1)
                positive_transh_score = transh_scores_keep[positive_x, positive_y]
                negative_transh_score = transh_scores_keep[negative_x, negative_y] 
                # positive_transh_head_entitys = transh_head_entitys_keep[positive_x, positive_y]
                # negative_transh_head_entitys = transh_head_entitys_keep[negative_x, negative_y]
                final_transh_head_entitys = torch.cat((transh_head_entitys_keep[positive_x, positive_y],transh_head_entitys_keep[negative_x, negative_y]),0)
                # positive_transh_tail_entitys = transh_tail_entitys_keep[positive_x, positive_y]
                # negative_transh_tail_entitys = transh_tail_entitys_keep[negative_x, negative_y]
                final_transh_tail_entitys = torch.cat((transh_tail_entitys_keep[positive_x, positive_y],transh_tail_entitys_keep[negative_x, negative_y]),0)
                # positive_transh_relations = transh_relations_keep[positive_x, positive_y]
                # negative_transh_relations = transh_relations_keep[negative_x, negative_y]
                final_transh_relations = torch.cat((transh_relations_keep[positive_x, positive_y],transh_relations_keep[negative_x, negative_y]),0)
                # positive_transh_relations_norm = transh_relations_norm_keep[positive_x, positive_y]
                # negative_transh_relations_norm = transh_relations_norm_keep[negative_x, negative_y]
                final_transh_relations_norm = torch.cat(( transh_relations_norm_keep[positive_x, positive_y],transh_relations_norm_keep[negative_x, negative_y]),0)
                
                all_labels.append(target_label)
                all_positive_transH_scores.append(positive_transh_score)
                all_negative_transH_scores.append(negative_transh_score)
                all_head_entitys.append(final_transh_head_entitys)
                all_tail_entitys.append(final_transh_tail_entitys)
                all_relations.append(final_transh_relations)
                all_relations_norm.append(final_transh_relations_norm)
                

            n_h_e = h_node_encodings[x_keep]
            n_e = node_encodings[y_keep]
            b_p_s_r = box_pair_spatial_reshaped[x_keep, y_keep]

            attention1 = self.attention_head(torch.cat([n_h_e,n_e], 1),b_p_s_r)
            attention2 = self.attention_head_g(global_features[b_idx, None],
                                               box_pair_spatial_reshaped[x_keep, y_keep])
            all_box_pair_features.append(torch.cat([attention1, attention2], dim=1))
            all_boxes_h.append(coords[x_keep])
            all_boxes_o.append(coords[y_keep])
            all_object_class.append(labels[y_keep])
            # The prior score is the product of the object detection scores
            all_prior.append(self.compute_prior_scores(x_keep, y_keep, scores, labels))



            counter += n
        if self.training:

            return all_box_pair_features, all_boxes_h, all_boxes_o, \
                    all_object_class, all_labels, all_prior, \
                    all_positive_transH_scores,all_negative_transH_scores,\
                    all_head_entitys,all_tail_entitys,all_relations,all_relations_norm
        else:

            return all_box_pair_features, all_boxes_h, all_boxes_o, \
                    all_object_class, all_labels, all_prior, \
                    all_positive_transH_scores,all_negative_transH_scores
            # return all_box_pair_features, all_boxes_h, all_boxes_o, \
            #            all_object_class, all_labels, all_prior




        