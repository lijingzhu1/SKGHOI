"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops
import os
import json
import pocket
from pocket.data import HICODet
from tqdm import tqdm
from torch.nn import Module
from torch import nn, Tensor
from pocket.ops import Flatten
from typing import Optional, List, Tuple
from collections import OrderedDict
import numpy as np
from torchvision.ops import MultiScaleRoIAlign
from transforms import HOINetworkTransform
import cv2
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

from ops import compute_spatial_encodings, binary_focal_loss
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger() 

def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    # xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin, ymin, xmax, ymax = boxes
    # logger.info(f'xmin, ymin, xmax, ymax:{xmin}, {ymin}, {xmax}, {ymax}')

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=0)

def model_preprocess(
    images: List[Tensor],
    detections: List[dict],
    targets: Optional[List[dict]] = None
    ) -> Tuple[
        List[dict],
        List[dict], List[Tuple[int, int]]
]:
    original_image_sizes = [img.shape[-2:] for img in images]
    # logger.info(f'original_image_sizes:{original_image_sizes}')
    images = images.unsqueeze_(0)
    # logger.info(f'images:{images.size( )}')
    images = HOI_transform(images)
    # logger.info(f'detections:{detections.items()}')

    for det, o_im_s, im_s in zip(detections['boxes'], original_image_sizes, images[0].image_sizes):
        # logger.info(f'det before resize:{det}')
        # boxes = det['boxes']
        boxes = resize_boxes(det, o_im_s, im_s)
        det = boxes
    # logger.info(f'det after resize:{det}')
    return images, detections, original_image_sizes

def head_preprocess(
    detections: List[dict],
    append_gt: Optional[bool] = None
) -> None:

results = []
    # for b_idx,(boxes,labels,scores) in enumerate(zip(detections['boxes'],detections['labels'],detections['scores'])):
        # boxes = detection['boxes']
        # labels = detection['labels']
        # scores = detection['scores']

        # Append ground truth during training
        # if append_gt is None:
        #     append_gt = self.training
        # if append_gt:
        #     target = targets[b_idx]
        #     n = target["boxes_h"].shape[0]
        #     boxes = torch.cat([target["boxes_h"], target["boxes_o"], boxes])
        #     scores = torch.cat([torch.ones(2 * n, device=scores.device), scores])
        #     labels = torch.cat([
        #         49 * torch.ones(n, device=labels.device).long(),
        #         target["object"],
        #         labels
        #     ])
    logger.info(f'scores:{scores}')
    # Remove low scoring examples
    active_idx = torch.nonzero(
        scores >= 0.2
    ).squeeze(1)
    # Class-wise non-maximum suppression
    keep_idx = box_ops.batched_nms(
        boxes[active_idx],
        scores[active_idx],
        labels[active_idx],
        0.5
    )
    active_idx = active_idx[keep_idx]
    # Sort detections by scores
    sorted_idx = torch.argsort(scores[active_idx], descending=True)
    active_idx = active_idx[sorted_idx]
    # Keep a fixed number of detections
    h_idx = torch.nonzero(labels[active_idx] == 49).squeeze(1)
    o_idx = torch.nonzero(labels[active_idx] != 49).squeeze(1)
    if len(h_idx) > 15:
        h_idx = h_idx[:15]
    if len(o_idx) > 15:
        o_idx = o_idx[:15]
    # Permute humans to the top
    keep_idx = torch.cat([h_idx, o_idx])
    active_idx = active_idx[keep_idx]

    results.append(dict(
        boxes=boxes[active_idx].view(-1, 4),
        labels=labels[active_idx].view(-1),
        scores=scores[active_idx].view(-1)
    
    ))

return results

cache_dir = os.path.join('/users/PCS0256/lijing/spatially-conditioned-graphs/hicodet/detections/', 'train2015')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

dataset = HICODet(
    root=os.path.join('/users/PCS0256/lijing/spatially-conditioned-graphs/hicodet/',
        "hico_20160224_det/images/{}".format('train2015')),
    anno_file=os.path.join('/users/PCS0256/lijing/spatially-conditioned-graphs/hicodet/',
        "instances_{}.json".format('train2015'))
)
config = '/users/PCS0256/lijing/mmdetection_ascend/configs/adamixer/adamixer_hicodetOD.py'
checkpoint = '/users/PCS0256/lijing/mmdetection_ascend/checkpoints/hoi_adamixer/adamixer_hicodetOD/epoch_12.pth'

cfg = mmcv.Config.fromfile(config)
detector = build_detector(cfg['model'])
if checkpoint is not None:
    checkpoint = load_checkpoint(detector, checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        detector.CLASSES = checkpoint['meta']['CLASSES']
    else:
        import warnings
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use COCO classes by default.')
        detector.CLASSES = get_classes('coco')

detector.eval()
detector.cuda()
is_batch = False

detector_backbone = detector.backbone
detector_neck = detector.neck
HOI_transform = HOINetworkTransform(800, 1333,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])



box_roi_pool = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2
)
# detections = []
for idx, (image, _) in enumerate(tqdm(dataset)):
    images = cv2.imread(os.path.join('/users/PCS0256/lijing/spatially-conditioned-graphs/hicodet/',
        "hico_20160224_det/images/{}".format('train2015'), dataset.filename(idx)))
    detection_path = os.path.join('/users/PCS0256/lijing/spatially-conditioned-graphs/hicodet/detections/train2015/',dataset.filename(idx).replace('jpg', 'json'))
    with open(detection_path, 'r') as f:
        detections = pocket.ops.to_tensor(json.load(f),input_format='dict')
    # images = np.transpose(images, (2,0,1))
    # logger.info(f'images:{images.shape}')
    images = torchvision.transforms.functional.to_tensor(images)
    # images = torch.from_numpy(images)


    # detections = torch.from_numpy(detections)
    images, detections,original_image_sizes = model_preprocess(images, detections)
    logger.info(f'detections:{detections}')
    detections = head_preprocess(detections)
    features = detector_backbone(images.tensors)
    features = detector_neck(features)
    box_feature = OrderedDict()
    box_feature['0'] = features[0]
    box_feature['1'] = features[1]
    box_feature['2'] = features[2]
    box_feature['3'] = features[3]
    box_coords = detections['boxes']
    box_labels = detections['labels']
    box_scores = detections['scores']
    box_features = box_roi_pool(features, box_coords, original_image_sizes)
    # entity = torch.stack(box_features)
    logger.info(f'box_features:{box_features.size()}')
    


