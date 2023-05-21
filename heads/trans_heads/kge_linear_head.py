

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
import itertools

from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict
import random
import numpy as np
torch.cuda.empty_cache()

from ops import compute_spatial_encodings, binary_focal_loss
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger()  
import clip

import sys
sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs/heads/TransH')
from LinearH import LinearH
sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs/heads')
from NegativeSampling import NegativeSampling
from MarginLoss import MarginLoss

def choose_random_number(min_value, max_value, exclude_list):
    while True:
        random_number = random.randint(min_value, max_value)
        if random_number not in exclude_list:
            return random_number

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
        # box_pair_suppressor: Module,
        # box_pair_predictor: Module,
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
        # self.box_pair_suppressor = box_pair_suppressor
        # self.box_pair_predictor = box_pair_predictor

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
        scores:List[Tensor],
        # negative_scores:List[Tensor],
        # head:List[Tensor],
        # relation:List[Tensor],
        # relation_norm:List[Tensor],
        # tail:List[Tensor],
        # results: List[dict]
    ) -> Tensor:
	    # logger.info(f'scores:{scores}')
	    n_p = len(scores)
	    
	    model = NegativeSampling(loss=MarginLoss(margin=4.0), batch_size=256).to('cuda')
	    if self.distributed:
	        world_size = dist.get_world_size()
	        n_p = torch.as_tensor([n_p], device='cuda')
	        dist.barrier()
	        dist.all_reduce(n_p)
	        n_p = (n_p / world_size).item()
	    losses = 0
	    for i in scores:
	    	# logger.info(f'i:{i}')
	    	loss = model(i)
	    	losses+=loss


	    final_loss = losses/ n_p
	    logger.info(f'final_loss:{final_loss}')
	    return final_loss

    # def postprocess(self,
    #     logits_p: Tensor,
    #     logits_s: Tensor,
    #     prior: List[Tensor],
    #     boxes_h: List[Tensor],
    #     boxes_o: List[Tensor],
    #     object_class: List[Tensor],
    #     labels: List[Tensor]
    # ) -> Tuple[
    #     List[dict]
    #     ]:
    #     # ) -> List[dict]:
    #     """
    #     Parameters:
    #     -----------
    #     logits_p: Tensor
    #         (N, K) Classification logits on each action for all box pairs
    #     logits_s: Tensor
    #         (N, 1) Logits for unary weights
    #     prior: List[Tensor]
    #         Prior scores organised by images. Each tensor has shape (2, M, K).
    #         M could be different for different images
    #     boxes_h: List[Tensor]
    #         Human bounding box coordinates organised by images (M, 4)
    #     boxes_o: List[Tensor]
    #         Object bounding box coordinates organised by images (M, 4)
    #     object_classes: List[Tensor]
    #         Object indices for each pair organised by images (M,)
    #     labels: List[Tensor]
    #         Binary labels on each action organised by images (M, K)

    #     Returns:
    #     --------
    #     results: List[dict]
    #         Results organised by images, with keys as below
    #         `boxes_h`: Tensor[M, 4]
    #         `boxes_o`: Tensor[M, 4]
    #         `index`: Tensor[L]
    #             Expanded indices of box pairs for each predicted action
    #         `prediction`: Tensor[L]
    #             Expanded indices of predicted actions
    #         `scores`: Tensor[L]
    #             Scores for each predicted action
    #         `object`: Tensor[M]
    #             Object indices for each pair
    #         `prior`: Tensor[2, L]
    #             Prior scores for expanded pairs
    #         `weights`: Tensor[M]
    #             Unary weights for each box pair
    #         `labels`: Tensor[L], optional
    #             Binary labels on each action
    #         `unary_labels`: Tensor[M], optional
    #             Labels for the unary weights
    #     """
    #     num_boxes = [len(b) for b in boxes_h]
    #     # logger.info(f'transh_score:{transh_score}')
    #     weights = torch.sigmoid(logits_s).squeeze(1)
    #     # logger.info(f'min of x: {min(x)}')
    #     scores = torch.sigmoid(logits_p)
    #     weights = weights.split(num_boxes)
    #     scores = scores.split(num_boxes)
    #     if len(labels) == 0:
    #         labels = [None for _ in range(len(num_boxes))]
    #     # if len(transH_positive_scores) == 0:
    #     #     transH_positive_scores = [None for _ in range(len(num_boxes))]
    #     # if len(transH_negative_scores) == 0:
    #     #     transH_negative_scores = [None for _ in range(len(num_boxes))]

    #     results = []
    #     # transH_positive_score_list = []
    #     # transH_negative_score_list = []
    #     for w, s, p, b_h, b_o, o, l in zip(
    #         weights, scores, prior, boxes_h, boxes_o, object_class, labels
    #     ):
    #     # for w, s, p, b_h, b_o, o, l in zip(
    #     #     weights, scores, prior, boxes_h, boxes_o, object_class, labels
    #     #     ):
    #         # Keep valid classes
    #         x, y = torch.nonzero(p[0]).unbind(1)
    #         action_score = s[x, y] * p[:, x, y].prod(dim=0) * w[x].detach()
    #         result_dict = dict(
    #             boxes_h=b_h, boxes_o=b_o,
    #             index=x, prediction=y,
    #             scores=action_score,
    #             object=o, prior=p[:, x, y], weights=w
    #         )
    #         # logger.info(f'results:{result_dict}')
    #         # If binary labels are provided
    #         if l is not None:
    #             ##one to many verb prediction
    #             action_binary_label = l[x, y] 
    #             result_dict['labels'] = action_binary_label 
    #             ##whether pair is valid
    #             result_dict['unary_labels'] = l.sum(dim=1).clamp(max=1)
 

    #         results.append(result_dict)
    #         # transH_positive_score_list.append(t_p)
    #         # transH_negative_score_list.append(t_n)

    #     return results
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
        results =[]
        if self.training:
        	scores = self.box_pair_head(features, image_shapes, box_features,\
										box_coords, box_labels, box_scores, targets)
        	loss_dict = dict(transH_loss = self.compute_transH_loss(scores))
        	results.append(loss_dict)

        else:
            box_pair_features, boxes_h, boxes_o, object_class,\
            box_pair_labels, box_pair_prior,transH_positive_scores,\
            transH_negative_scores = self.box_pair_head(
                features, image_shapes, box_features,
                box_coords, box_labels, box_scores, targets)

        return results


class LinearhHead(Module):

    def __init__(self,
    	# entity_size:int,
    	# relation_size:int,
        entity_encoding_size: int,
        relation_encoding_size_1:int,
        relation_encoding_size_2:int,
        trans_dim: int,
        transh_p_norm: int,
        transh_norm_flag: bool,
        human_idx: int, 
        num_object: int,
        num_cls: int,
    ) -> None:

        super().__init__()
        self.trans_dim = trans_dim
        # self.entity_size = entity_size
        # self.relation_size = relation_size
        self.entity_encoding_size = entity_encoding_size
        self.relation_encoding_size_1 = relation_encoding_size_1
        self.relation_encoding_size_2 = relation_encoding_size_2
        self.transh_p_norm = transh_p_norm
        self.transh_norm_flag = transh_norm_flag
        self.device = 'cuda'
        self.human_idx = human_idx
        self.num_object = num_object
        self.num_cls = num_cls


    def forward(self,
        head:Tensor,
        relation: Tensor,
        tail: Tensor,
        ) -> Tuple[
        Tensor, Tensor,Tensor,Tensor,Tensor
        ]:

        # relations = torch.tensor([*range(0,self.num_cls)],device=self.device, dtype=torch.int64).repeat(len(ind_x))
        # heads = torch.tensor([self.human_idx],device=self.device, dtype=torch.int64).repeat(len(ind_x)*self.num_cls)
        # tails = torch.tensor(labels.repeat_interleave(self.num_cls),device=self.device, dtype=torch.int64).repeat(n_h)

        linearh= LinearH(ent_tot = 10, rel_tot = 10, entity_size = 1024, relation_size = 1024,\
        				entity_encoding_size = self.entity_encoding_size,relation_encoding_size_1 = self.relation_encoding_size_1,\
        				relation_encoding_size_2 = self.relation_encoding_size_2,trans_dim = self.trans_dim,\
                        p_norm = self.transh_p_norm,norm_flag = self.transh_norm_flag).to(self.device)
        # all_head = 
        # all_tail = torch.cat((tail,head),dim = 0)
        # all_relation = torch.cat((relation,relation),dim = 0)
        # if relation is not None:
        score = linearh(head,relation,tail)
        # logger.info(f'head_entity,tail_entity,transh_scores:{head_entity.size()},{tail_entity.size()},{transh_scores.size()}')

        return score



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
        num_iter: int = 2,
        trans_head = Module,
        obj_text_label = list,
        hoi_text_label = dict,
        hoi_valid_pair = dict,
        hoi_action_ongoing_label = list,
        hoi_action_valid_object = dict,
        clip_model = str,
        spatial_encoded_dim = int
    ) -> None:

        super().__init__()

        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.human_idx = human_idx
        # self.num_obj = num_objects
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter
        self.trans_head = trans_head
        self.obj_text_label = obj_text_label
        self.hoi_text_label = hoi_text_label
        self.hoi_valid_pair = hoi_valid_pair
        self.hoi_action_ongoing_label = hoi_action_ongoing_label
        self.hoi_action_valid_object = hoi_action_valid_object
        self.clip_model = clip_model

        self.appearance_feature_project_head = nn.Sequential(
            Flatten(start_dim=1),
            nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
           	# nn.BatchNorm1d(node_encoding_size),
            nn.ReLU(),
            nn.Linear(node_encoding_size, node_encoding_size),
            # nn.BatchNorm1d(node_encoding_size),
            nn.ReLU())

        self.spatial_head = nn.Sequential(
		            nn.Linear(36, 128),
		            # nn.BatchNorm1d(128),
		            nn.ReLU(),
		            nn.Linear(128, 256),
		            # nn.BatchNorm1d(256),
		            nn.ReLU(),
		            nn.Linear(256, spatial_encoded_dim),
		            # nn.BatchNorm1d(spatial_encoded_dim),
		            nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)


    def associate_with_ground_truth(self,
        boxes_h: Tensor,
        boxes_o: Tensor,
        targets: List[dict]
    ) -> Tensor:
        n = boxes_h.shape[0]
        # labels = torch.zeros(n, self.num_cls, device=boxes_h.device)
        # logger.info(f'targets:{targets}')

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)
 
        # labels[x, targets["labels"][y]] = 1
        # logger.info(f'targets["labels"]:{targets["labels"]}')
        # verb_label = targets["labels"][y]
        # verb_label = labels[x, targets["labels"][y]]
        # logger.info(f'verb_label:{verb_label}')
        return x,y,targets["labels"]


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
        # logger.info(f'box_features:{box_features.size()}')
        # box_features = self.box_head(box_features)

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        # all_positive_transH_scores = []; all_negative_transH_scores = []
        all_scores = []
        # all_head_entitys = []; all_tail_entitys = []; all_relations = [];all_relations_norm = []
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
                # all_prior.append(torch.zeros(2, 0, self.num_cls, device=device))
                all_labels.append(torch.zeros(0, self.num_cls, device=device))
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise ValueError("Human detections are not permuted to the top")

            appearance_feature = box_features[counter: counter+n]
            # Duplicate human nodes
            h_appearance_feature = appearance_feature[:n_h] 
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
            # Remove pairs consisting of the same human instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            # logger.info(f'labels:{labels}')
            # logger.info(f'x_keep, y_keep:{x_keep},{y_keep}')
            
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()
            # logger.info(f'x, y:{x},{y}')

            #Flatten and project appearance feature
            # logger.info(f'h_appearance_feature[x_keep]:{h_appearance_feature[x_keep].size()}')
            human_appearance_feature = self.appearance_feature_project_head(h_appearance_feature[x_keep])
            # logger.info(f'human_appearance_feature:{human_appearance_feature.size()}')
            object_appearance_feature = self.appearance_feature_project_head(appearance_feature[y_keep])

            # logger.info(f'human_appearance_feature:{human_appearance_feature.size()}')
            obj_text_labels = [self.obj_text_label[label] for label in labels]
            obj_text_sentences = [(str('A photo of an') + ' '+ str(obj_text_label)) \
            							if obj_text_label[0] in ('a','e','i','o','u') \
            								else (str('A photo of a') + ' '+ str(obj_text_label)) \
                                        for obj_text_label in obj_text_labels]
            # logger.info(f'obj_text_sentences:{obj_text_sentences}')
            obj_text_inputs = clip.tokenize(obj_text_sentences)
            # logger.info(f'obj_text_inputs:{obj_text_inputs.size()}')
            clip_model, preprocess = clip.load(self.clip_model, device=device)
            for param in clip_model.parameters():
            	param.requires_grad = False
            with torch.no_grad():
            	obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            del clip_model
            	# logger.info(f'obj_text_embedding:{obj_text_embedding.size()}')
            human_entity_embedding = torch.cat((human_appearance_feature,obj_text_embedding[x_keep]),1)
            object_entity_embedding = torch.cat((object_appearance_feature,obj_text_embedding[y_keep]),1)

            if targets is not None:
            	# Compute spatial features
            	box_pair_spatial = compute_spatial_encodings([coords[x_keep]], [coords[y_keep]], [image_shapes[b_idx]])
            	# logger.info(f'box_pair_spatial:{box_pair_spatial.size()}')
            	if torch.isnan(box_pair_spatial).sum() > 0:
            		logger.info(f'produced the box_parir_spatial nan values: {box_pair_spatial}')
            		box_pair_spatial = torch.nan_to_num(box_pair_spatial)
            	#Project spatial features
            	box_spatial_project = self.spatial_head(box_pair_spatial)
            	# logger.info(f'box_spatial_project:{box_spatial_project.size()}')
            	pair_x_keep,pair_y_keep,action_label = self.associate_with_ground_truth(coords[x_keep], coords[y_keep], targets[b_idx])

            	# logger.info(f'pair_x_keep,pair_y_keep,action_label:{pair_x_keep},{pair_y_keep},{action_label}')
            	if len(pair_x_keep) < 1:
            		pass
            	else:

	            	box_spatial_project_keep = box_spatial_project[pair_x_keep]
	            	# logger.info(f'box_spatial_project_keep:{box_spatial_project_keep.size()}')
	            	action_label_keep = action_label[pair_y_keep]
	            	# logger.info(f'labels[x_keep]:{labels[y_keep]}')
	            	all_positive_hoi_text_sentence = []; all_negative_hoi_text_sentece = [];all_negative_obj_text_sentence=[]
	            	all_box_spatial_project_keep = []
	            	for action_label, object_label, box_spatial in zip(action_label_keep,labels[y_keep][pair_x_keep],box_spatial_project_keep):

	            		# logger.info(f'action_label:{action_label}')
	            		# logger.info(f'object_label:{object_label}')
	            		unvalid_hoi_pair = [num for num in range(80) if num not in self.hoi_valid_pair[action_label.item()]]
	            		# logger.info(f'length of unvalid_hoi_pair:{len(unvalid_hoi_pair)}')
	            		if len(unvalid_hoi_pair) < 1:
	            			# logger.info(f'self.hoi_action_valid_object:{self.hoi_action_valid_object}')
	            			valid_object_list = self.hoi_action_valid_object[object_label.item()]
	            			# logger.info(f'valid_object_list:{valid_object_list}')
	            			random_action_label = choose_random_number(0,116,valid_object_list)
	            			# logger.info(f'random_action_label:{random_action_label}')
	            			negative_action_text_label = self.hoi_action_ongoing_label[random_action_label]
	            			# logger.info(f'negative_action_text_label:{negative_action_text_label}')
	            			# negative_object_text_sentences = [(str('A photo of an') + ' '+ str(self.obj_text_label[negative_object_random_number])) \
	            			# 				if str(object_label)[0] in ('a','e','i','o','u') \
	            			# 					else (str('A photo of a') + ' '+ str(object_label))]
	            			obj_text_label = self.obj_text_label[object_label]
	            			# logger.info(f'obj_text_label:{obj_text_label}')
	            			negative_object_text_sentences = (str('A photo of an') + ' '+ str(obj_text_label)) \
	            							if obj_text_label[0] in ('a','e','i','o','u') \
	            								else (str('A photo of a') + ' '+ str(obj_text_label))
	            			negative_hoi_sentence = (str('A photo of a person') + ' '+ \
	            												str(negative_action_text_label)) +' ' +str('an') + ' '+str(obj_text_label)\
	            							if str(obj_text_label)[0] in ('a','e','i','o','u') \
	            							else (str('A photo of a person') + ' '+ \
	            												str(negative_action_text_label)) +' '+ str('a') + ' '+str(obj_text_label)

	            		else:
	            			negative_object_random_number = random.choice(unvalid_hoi_pair)
	            			# logger.info(f'negative_object_random_number:{negative_object_random_number}')
	            			negative_object_text_label = self.obj_text_label[negative_object_random_number]
	            			negative_object_text_sentences = (str('A photo of an') + ' '+ str(negative_object_text_label)) \
	            							if negative_object_text_label[0] in ('a','e','i','o','u') \
	            								else (str('A photo of a') + ' '+ str(negative_object_text_label))
	            		
	            			hoi_pair = tuple((action_label.item(),object_label.item()))
	            			# logger.info(f'hoi_pair:{hoi_pair}')
	            			hoi_text_sentences = self.hoi_text_label.get(hoi_pair)
	            			# logger.info(f'hoi_text_sentences:{hoi_text_sentences}')
	            			if hoi_text_sentences is not None:
		            			# logger.info(f'hoi_text_sentences:{hoi_text_sentences}')
		            			sentence = hoi_text_sentences.split(' ')
		            			#find the last index of 'a' or 'an'
		            			index = max([i for i, word in enumerate(sentence) if word in ['a', 'an']])
		            			# extract the sentence before the last 'a' or 'an'
		            			extracted_sentence = ' '.join(sentence[:index])
		            			# logger.info(f'extracted_sentence:{extracted_sentence}')
		            			negative_hoi_sentence = (str(extracted_sentence)+ ' '+ str('an')+ ' '+ str(negative_object_text_label)) \
		            							if negative_object_text_label[0] in ('a','e','i','o','u') \
		            							else (str(extracted_sentence)+ ' '+ str('a')+ ' '+ str(negative_object_text_label))
		            		else:
		            			hoi_text_sentences  = None
		            			negative_hoi_sentence = None
		            			negative_object_text_sentences =  None
		            			box_spatial = None

	            		# logger.info(f'negative_object_text_sentences:{negative_object_text_sentences}')
	            		# logger.info(f'negative_hoi_sentence:{negative_hoi_sentence}')
	            		if hoi_text_sentences is not None:
	            			all_positive_hoi_text_sentence.append(hoi_text_sentences)
	            		if negative_hoi_sentence is not None:
		            	# logger.info(f'all_hoi_text_sentences:{all_hoi_text_sentence}')
		            		all_negative_hoi_text_sentece.append(negative_hoi_sentence)
		            	if negative_object_text_sentences is not None:
		            		all_negative_obj_text_sentence.append(negative_object_text_sentences)
		            	if box_spatial is not None:
		            		all_box_spatial_project_keep.append(box_spatial)
            	# logger.info(f'all_negative_hoi_text_sentece:{all_negative_hoi_text_sentece}')
	            	positive_hoi_text_inputs = clip.tokenize(all_positive_hoi_text_sentence)
	            	negative_hoi_text_inputs = clip.tokenize(all_negative_hoi_text_sentece)
	            	negative_obj_text_inputs = clip.tokenize(all_negative_obj_text_sentence)
            	# negative_object_text_label
            		clip_model, preprocess = clip.load(self.clip_model, device=device)
	            	for param in clip_model.parameters():
	            		param.requires_grad = False
	            	with torch.no_grad():
	            		positive_hoi_text_embedding = clip_model.encode_text(positive_hoi_text_inputs.to(device))
	            		negative_obj_text_embedding = clip_model.encode_text(negative_obj_text_inputs.to(device))
	            		negative_hoi_text_embedding = clip_model.encode_text(negative_hoi_text_inputs.to(device))
	            		# logger.info(f'positive_hoi_text_embedding:{positive_hoi_text_embedding.size()}')
	            	del clip_model
	            	positive_relation_embedding = torch.cat((all_box_spatial_project_keep,positive_hoi_text_embedding),1)
	            	# negative_box_spatial_size = 
	            	negative_box_spatial = torch.abs(torch.randn((all_box_spatial_project_keep.size()[0],512))).to(device)
	            	negative_object_appearance_feature =  torch.abs(torch.randn((object_appearance_feature[pair_x_keep].size()[0],512))).to(device)
	            	# logger.info(f'negative_box_spatial_size:{negative_box_spatial.size()}')
	            	# logger.info(f'negative_object_appearance_feature:{negative_object_appearance_feature.size()}')
	            	negative_relation_embedding = torch.cat((negative_box_spatial,negative_hoi_text_embedding),1)
	            	# logger.info(f'negative_relation_embedding:{negative_relation_embedding.size()}')
	            	positive_head_embedding = human_entity_embedding[pair_x_keep]
	            	positive_tail_embedding = object_entity_embedding[pair_x_keep]
	            	negative_tail_embedding = torch.cat((negative_object_appearance_feature,negative_obj_text_embedding),1)
	            	all_relation_embedding = torch.cat((positive_relation_embedding,negative_relation_embedding),0)
	            	all_head_embedding = positive_head_embedding.repeat(2,1)
	            	all_tail_embedding = torch.cat((positive_tail_embedding,negative_tail_embedding),0)
            	# logger.info(f'all_tail_embedding:{all_tail_embedding.size()}')
            	# logger.info(f'all_relation_embedding:{all_relation_embedding.size()}')
            	# logger.info(f'all_head_entity_embedding:{all_head_embedding.size()}')
            	

            		score = self.trans_head(all_head_embedding,all_relation_embedding,all_tail_embedding)
            		all_scores.append(score)
            	# positive_x, positive_y = torch.nonzero(target_label).unbind(1)
                # logger.info(f'positive_x, positive_y:{positive_x.size()}, {positive_y.size()}')
                # unique_x = torch.unique(positive_x,dim = 0)
                # transh_positive_score = transh_positive_scores_keep[unique_x]
                # transh_negative_score = transh_negative_scores_keep[unique_x]
                # final_transh_head_entitys = torch.cat((transh_positive_head_entitys_keep[unique_x],transh_negative_head_entitys_keep[unique_x]),0)
                # # logger.info(f'final_transh_head_entitys:{final_transh_head_entitys.size()}')
                # final_transh_tail_entitys = torch.cat((transh_positive_tail_entitys_keep[unique_x],transh_negative_tail_entitys_keep[unique_x]),0)
                # final_transh_relations = torch.cat((transh_positive_relations_keep[unique_x],transh_negative_relations_keep[unique_x]),0)
                # # positive_transh_relations_norm = transh_relations_norm_keep[positive_x, positive_y]
                # # negative_transh_relations_norm = transh_relations_norm_keep[negative_x, negative_y]
                # final_transh_relations_norm = torch.cat((transh_positive_relations_norm_keep[unique_x],transh_negative_relations_norm_keep[unique_x]),0)
                
                # all_labels.append(target_label)

                # all_negative_transH_scores.append(negative_transh_score)
                # all_head_entitys.append(final_transh_head_entitys)
                # all_tail_entitys.append(final_transh_tail_entitys)
                # all_relations.append(final_transh_relations)
                # all_relations_norm.append(final_transh_relations_norm)
                


            counter += n
        if self.training:

            return all_scores
        else:

            return all_positive_transH_scores,all_negative_transH_scores
            # return all_box_pair_features, all_boxes_h, all_boxes_o, \
            #            all_object_class, all_labels, all_prior




        