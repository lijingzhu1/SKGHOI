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


from ops import compute_spatial_encodings, binary_focal_loss
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger()  

import sys
sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs/heads')
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
        box_pair_head: Module,
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

        self.box_pair_head = box_pair_head

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
 

    def compute_transH_loss(self, 
        positive_scores:List[Tensor],
        negative_scores:List[Tensor],
        head:List[Tensor],
        relation:List[Tensor],
        relation_norm:List[Tensor],
        tail:List[Tensor],
        transH:Module
    ) -> Tensor:

        model = NegativeSampling(
            model = transH, 
            loss = MarginLoss(margin = 0.1),
            batch_size = 256,
            regul_rate = 0.25
        ).to('cuda')
        loss = model(positive_scores,negative_scores,head,relation,relation_norm,tail)
        # logger.info(f'transH_loss:{loss}')
        return loss

    def postprocess(self,
        transH_positive_scores:List[Tensor],
        transH_negative_scores:List[Tensor]
    ) -> Tuple[
        List[Tensor],List[Tensor]
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
        # num_boxes = [len(b) for b in boxes_h]
        # logger.info(f'transh_score:{transh_score}')

        if len(transH_positive_scores) == 0:
            transH_positive_scores = [None for _ in range(len(num_boxes))]
        if len(transH_negative_scores) == 0:
            transH_negative_scores = [None for _ in range(len(num_boxes))]
        transH_positive_score_list = []
        transH_negative_score_list = []
        for t_p, t_n in zip( transH_positive_scores,transH_negative_scores):
            transH_positive_score_list.append(t_p)
            transH_negative_score_list.append(t_n)

        return transH_positive_score_list,transH_negative_score_list
        # return results


    def forward(self,
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
        # logger.info(f'box_features:{box_features}')
        if self.training:
            transH_positive_scores,transH_negative_scores,head_entitys,relations,relations_norm,tail_entitys,transH = self.box_pair_head(
               box_coords, box_labels, box_scores, targets
            )
            head_entitys = torch.cat(head_entitys)
            tail_entitys = torch.cat(tail_entitys)

        else:
            transH_positive_scores, transH_negative_scores = self.box_pair_head(
                features, image_shapes, box_features,
                box_coords, box_labels, box_scores, targets)
        # box_pair_features, boxes_h, boxes_o, object_class, box_pair_labels, box_pair_prior = self.box_pair_head(

        ##logits_p,logits_s: predictors
        # boxes_h,boxes_o, box_pair_prior, box_pair_labels,object_class: targets
        transh_positive_scores_list,transh_negative_scores_list = self.postprocess(
            transH_positive_scores,transH_negative_scores
        )
        
        results = []
        if self.training:
            loss_dict = dict(
                transH_loss = self.compute_transH_loss(transh_positive_scores_list,transh_negative_scores_list,head_entitys,relations,relations_norm,tail_entitys,transH)
            )
            # logger.info(f'compute interactiveness loss:{loss_dict.values()}')
            results.append(loss_dict)

        return results


class transH_head(Module):

    def __init__(self,
        transh_dim: int = 300,
        transh_p_norm: int = 2,
        transh_norm_flag: bool = True,
        human_idx: int = 49, 
        num_object: int = 80,
        num_cls: int = 117
    ) -> None:

        super().__init__()
        self.transh_dim = transh_dim
        self.transh_p_norm = transh_p_norm
        self.transh_norm_flag = transh_norm_flag
        self.device = 'cuda'
        self.human_idx = human_idx
        self.num_object = num_object
        self.num_cls = num_cls


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

        relations = torch.tensor([*range(0,self.num_cls)],device=self.device, dtype=torch.int64).repeat(len(ind_x)) #N*117
        # relations = relations.reshape(n_h,n,self.num_cls)[x_keep,y_keep]
        heads = torch.tensor([self.human_idx],device=self.device, dtype=torch.int64).repeat(len(ind_x)*self.num_cls) #N*117
        # heads = heads.reshape(n_h,n,self.num_cls)[x_keep,y_keep]
        tails = torch.tensor(labels.repeat_interleave(self.num_cls),device=self.device, dtype=torch.int64).repeat(n_h)
        # tails = tails.reshape(n_h,n,self.num_cls)[x_keep,y_keep] #N*117
        transH= TransH(ent_tot = self.num_object,
                        rel_tot = self.num_cls, 
                        dim = self.transh_dim, 
                        p_norm = self.transh_p_norm, 
                        norm_flag = self.transh_norm_flag).to(self.device)
        transH.save_checkpoint('/users/PCS0256/lijing/spatially-conditioned-graphs/checkpoints/transh.ckpt')
            
        head_entity,relation,relation_norm,tail_entity,transh_scores = transH(heads,relations,tails)
        # logger.info(f'head_entity,relatin,tail_entity,transh_scores:{head_entity.size()},{relation.size()},{tail_entity.size()},{transh_scores.size()}')

        # logger.info(f'transh_tail_entity 0 and -1:{transh_tail_entitys[0]},{transh_tail_entitys[-1]}')
        # transh_head_entitys_flatten = torch.flatten(transh_head_entitys_stack, start_dim=1,end_dim=-1)
        # transh_tail_entitys_flatten = torch.flatten(transh_tail_entitys_stack, start_dim=1,end_dim=-1)
        # transh_head_entitys_fc = self.fc_head(transh_head_entitys_flatten)
        # transh_tail_entitys_fc  = self.fc_tail(transh_tail_entitys_flatten)
        # logger.info(f'transh_socres:{transh_scores}')

        return head_entity,relation,relation_norm,tail_entity,transh_scores,transH



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
        num_cls: int, human_idx: int,
        object_class_to_target_class: List[list],
        fg_iou_thresh: float = 0.5,
        num_iter: int = 2
    ) -> None:

        super().__init__()

        self.num_cls = num_cls
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Box head to map RoI features to low dimensional
        self.transh_head = transH_head(
            transh_dim = 300,
            transh_p_norm = 2,
            transh_norm_flag = True,
            human_idx = self.human_idx,
            num_object = 80,
            num_cls = self.num_cls
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


    def forward(self,
        box_coords: List[Tensor],
        box_labels: List[Tensor], box_scores: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor],List[Tensor]

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

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        all_positive_transH_scores = []; all_negative_transH_scores = []; all_box_pair_features = []
        all_head_entitys = []; all_tail_entitys = []; all_labels = []; all_relations = [];all_relations_norm = []
        ### singer image
        for b_idx, (coords, labels, scores) in enumerate(zip(box_coords, box_labels, box_scores)):
            n = num_boxes[b_idx] #3
            # logger.info(f'box_features:{bbox_features}')
            device = 'cuda'
            n_h = torch.sum(labels == self.human_idx).item()
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                # all_boxes_h.append(torch.zeros(0, 4, device=device))
                # all_boxes_o.append(torch.zeros(0, 4, device=device))
                # all_object_class.append(torch.zeros(0, device=device, dtype=torch.int64))
                # all_prior.append(torch.zeros(2, 0, self.num_cls, device=device))
                all_labels.append(torch.zeros(0, self.num_cls, device=device))
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise ValueError("Human detections are not permuted to the top")
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
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
            ##graph embedding input

            transh_head_entitys,transh_relations,transh_relations_norm,transh_tail_entitys,transh_scores,transH = self.transh_head(
                                                        x,y,n_h,n,x_keep, y_keep,labels
                                                        )
            head_entitys,tail_entitys = [],[]
            for idx,(entity_x,entity_y) in enumerate(zip(transh_head_entitys,transh_tail_entitys)):   
                # transh_scores.append(transh_sore)
                if idx % self.num_cls == 0:
                    head_entitys.append(entity_x)
                    tail_entitys.append(entity_y)
            head_entitys_stack = torch.stack(head_entitys)
            tail_entitys_stack = torch.stack(tail_entitys)
            # logger.info(f'transH:{transH}')
            # logger.info(f'transh_head_entity,tail_entity:{transh_head_entitys.size()},{transh_tail_entitys.size()}')
            # logger.info(f'node_encodings:{h_node_encodings[x].size()},{node_encodings[y].size()}')
            # logger.info(f'graph_human_node,graph_object_node:{graph_human_node.size()},{graph_object_node.size()}')
            # Reshape the spatial features
            transh_scores_keep= transh_scores.reshape(n_h,n,self.num_cls)[x_keep,y_keep]
            transh_head_entitys_keep= transh_head_entitys.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            transh_tail_entitys_keep= transh_tail_entitys.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            transh_relations_keep = transh_relations.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            transh_relations_norm_keep = transh_relations_norm.reshape(n_h,n,self.num_cls,-1)[x_keep,y_keep]
            # logger.info(f'transh_head_entitys_keep:{transh_head_entitys_keep.size()}')
            # logger.info(f'transh_relations_keep:{transh_relations_keep.size()}')

            if targets is not None:
                target_label = self.associate_with_ground_truth(coords[x_keep], coords[y_keep], targets[b_idx])
                # logger.info(f'target_labels:{target_label.size()}')
                positive_x, positive_y = torch.nonzero(target_label).unbind(1)
                # logger.info(f'positive_x, positive_y:{positive_x}, {positive_y}')
                negative_xy = (target_label== 0).nonzero()
                rand_columns = torch.randperm(negative_xy.size()[0])[:len(positive_x)]
                negative_x,negative_y = negative_xy[rand_columns].unbind(1)
                positive_transh_score = transh_scores_keep[positive_x, positmive_y]
                negative_transh_score = transh_scores_keep[negative_x, negative_y] 
                positive_transh_head_entitys = transh_head_entitys_keep[positive_x, positive_y]
                negative_transh_head_entitys = transh_head_entitys_keep[negative_x, negative_y]
                final_transh_head_entitys = torch.cat((positive_transh_head_entitys,negative_transh_head_entitys),0)
                positive_transh_tail_entitys = transh_tail_entitys_keep[positive_x, positive_y]
                negative_transh_tail_entitys = transh_tail_entitys_keep[negative_x, negative_y]
                final_transh_tail_entitys = torch.cat((positive_transh_tail_entitys,negative_transh_tail_entitys),0)
                positive_transh_relations = transh_relations_keep[positive_x, positive_y]
                negative_transh_relations = transh_relations_keep[negative_x, negative_y]
                final_transh_relations = torch.cat((positive_transh_relations,negative_transh_relations),0)
                positive_transh_relations_norm = transh_relations_norm_keep[positive_x, positive_y]
                negative_transh_relations_norm = transh_relations_norm_keep[negative_x, negative_y]
                final_transh_relations_norm = torch.cat((positive_transh_relations_norm,negative_transh_relations_norm),0)
                
                all_labels.append(target_label)
                all_positive_transH_scores.append(positive_transh_score)
                    # logger.info(f'all_positive_transH_scores:{all_positive_transH_scores}')
                all_negative_transH_scores.append(negative_transh_score)
                    # logger.info(f'all_negative_transH_scores:{all_negative_transH_scores}')
                all_head_entitys.append(final_transh_head_entitys)
                all_tail_entitys.append(final_transh_tail_entitys)
                all_relations.append(final_transh_relations)
                all_relations_norm.append(final_transh_relations_norm)
            # logger.info(f'all_prior:{[12 for all_box_pair_feature in all_prior if (all_box_pair_feature is None)]}')
            # logger.info(f'length of all_boxes_o:{[[all_box_pair_feature.size()] for all_box_pair_feature in all_boxes_o]}')
            # logger.info(f'length of all_object_class:{[[all_box_pair_feature] for all_box_pair_feature in all_object_class]}')
            # logger.info(f'length of all_boxes_h:{[[all_box_pair_feature] for all_box_pair_feature in all_boxes_h]}')
            # logger.info(f'length of all_prior:{[[all_box_pair_feature] for all_box_pair_feature in all_prior]}')
            # logger.info(f'length of all_tail_entitys:{[[all_box_pair_feature] for all_box_pair_feature in all_tail_entitys]}')
            # logger.info(f'length of all_head_entitys:{[[all_box_pair_feature.size()] for all_box_pair_feature in all_head_entitys]}')
            # logger.info(f'length of all_prior:{[[all_box_pair_feature.size()] for all_box_pair_feature in all_prior]}')
            # logger.info(f'length of relations:{[[all_box_pair_feature.size()] for all_box_pair_feature in all_relations]}')
            # logger.info(f'all_label.size:{[label.size() for label in all_labels]}')
            # logger.info(f'all_positive_transH_scores,all_negative_transH_scores,all_labels:{all_positive_transH_scores},{all_negative_transH_scores},{all_labels}')
            # logger.info(f'transH:{transH.ent_embeddings}')


            counter += n
        if self.training:
            return all_positive_transH_scores,all_negative_transH_scores,all_head_entitys,all_relations,all_relations_norm,all_tail_entitys,transH
        else:
            return all_positive_transH_scores,all_negative_transH_scores
            # return all_box_pair_features, all_boxes_h, all_boxes_o, \
            #            all_object_class, all_labels, all_prior




        