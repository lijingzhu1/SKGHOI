import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

import torch
from torchvision.ops import nms
from pocket.data import HICODet
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger()

def visualize(args):
    # Set up root directory
    partition = args.partition
    dataset = HICODet(None, os.path.join(
        args.data_root, 'instances_{}.json'.format(args.partition)
    ))
    # Set up image instance path
    image_name = dataset.filename(args.image_idx)
    print("Image name: ", image_name)
    image_path = os.path.join(
        args.data_root,
        'hico_20160224_det/images/{}'.format(args.partition),
        image_name
    )
    detection_path = os.path.join(
        args.detection_root,
        image_name.replace('.jpg', '.json')
    )
    # Load image instance
    image = Image.open(image_path)
    with open(detection_path, 'r') as f:
        detections = json.load(f)
    # Remove low-scoring boxes
    box_score_thresh = args.box_score_thresh
    boxes = np.asarray(detections['boxes'])
    scores = np.asarray(detections['scores'])
    labels = np.asarray(detections['labels'])
    keep_idx = np.where(scores >= box_score_thresh)[0]
    boxes = boxes[keep_idx, :]
    scores = scores[keep_idx]
    labels = labels[keep_idx]
    # Perform NMS
    keep_idx = nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        args.nms_thresh
    )
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels= labels[keep_idx]

    CLASSES = ('airplane','apple','backpack','banana','baseball bat','baseball glove','bear','bed',
 'bench','bicycle','bird','boat','book','bottle','bowl','broccoli','bus','cake','car','carrot','cat',
 'cell phone','chair','clock','couch','cow','cup','dining table','dog','donut','elephant','fire hydrant',
 'fork','frisbee','giraffe','hair drier','handbag','horse','hot dog','keyboard','kite',
 'knife','laptop','microwave','motorcycle','mouse','orange','oven','parking meter','person',
 'pizza','potted plant','refrigerator','remote','sandwich','scissors','sheep','sink',
 'skateboard','skis','snowboard','spoon','sports ball','stop sign','suitcase','surfboard',
 'teddy bear','tennis racket','tie','toaster','toilet','toothbrush','traffic light','train',
 'truck','tv','umbrella','vase','wine glass','zebra')
    # Draw boxes
    labels = [CLASSES[label] for label in labels]
    canvas = ImageDraw.Draw(image)
    for idx in range(boxes.shape[0]):
        text = str(scores[idx])[:4]+' '+str(labels[idx])
        # logger.info(f'str(scores[idx])[:4]:{str(scores[idx])[:4]}')
        # logger.info(f'str(labels[idx])[:4]:{str(labels[idx])}')
        coords = boxes[idx, :].tolist()
        canvas.rectangle(coords)
        canvas.text(coords[:2],text)

    image.save(args.out_file)
    # mmcv.imwrite(image,args.out_file)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Visualize object detections")
    parser.add_argument('--detection-root', type=str, required=True)
    parser.add_argument('--image-idx', type=int, default=0)
    parser.add_argument('--out_file', type=str, default='result.jpg')
    parser.add_argument('--data-root', type=str, default='../')
    parser.add_argument('--partition', type=str, default='train2015')
    parser.add_argument('--box-score-thresh', type=float, default=0.3)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    args = parser.parse_args()

    visualize(args)
