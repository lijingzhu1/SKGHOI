"""
Run Faster R-CNN with ResNet50-FPN on HICO-DET

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import torch
import argparse
import torchvision
import cv2
from tqdm import tqdm
import numpy as np

from pocket.ops import relocate_to_cpu
from mmcv.parallel import collate, scatter
from pocket.data import HICODet
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
import mmcv
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger()

def main(args):
	cache_dir = os.path.join(args.cache_dir, 'test2015_r101_pretained')
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	with open(os.path.join(args.data_root, 'mmdet80tohico80.json'), 'r') as f:
		coco2hico = json.load(f)

	dataset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/{}".format(args.partition)),
        anno_file=os.path.join(args.data_root,
            "instances_{}.json".format(args.partition))
    )
	config = '/users/PCS0256/lijing/mmdetection_ascend/configs/adamixer/r101_36_epoch_finetuning_hicodet.py'
	checkpoint = '/users/PCS0256/lijing/mmdetection_ascend/checkpoints/hoi_adamixer/adamixer_finetuning_r101_36_epoch/epoch_27.pth'

	cfg = mmcv.Config.fromfile(config)
	detector = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

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

	for idx, (image, _) in enumerate(tqdm(dataset)):
		# image_path = img_metas[i].get('filename')
		# logger.info(f'image :{image}')
		# logger.info(f'_:{_}')
		img = cv2.imread(os.path.join(args.data_root,
            "hico_20160224_det/images/{}".format(args.partition), dataset.filename(idx)))
		# image = torchvision.transforms.functional.to_tensor(image).cuda()
		cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
		data = dict(img=img)
		cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
		test_pipeline = Compose(cfg.data.test.pipeline)
		data = test_pipeline(data)
		data = collate([data], samples_per_gpu=1)
		# just get the actual data from DataContainer
		data['img_metas'] = [
		    img_metas.data[0] for img_metas in data['img_metas']
		]
		data['img'] = [img.data[0] for img in data['img']]

		if next(detector.parameters()).is_cuda:
		    # scatter to specified GPU
		    data = scatter(data, ['cuda:0'])[0]
		else:
		    for m in detector.modules():
		        assert not isinstance(
		            m, RoIPool
		        ), 'CPU inference with RoIPool is not supported currently.'
		with torch.no_grad():
		    results = detector(return_loss=False, rescale=True, **data)

		if not is_batch:
		    bbox_result = results[0]
		else:
		    bbox_result =  results

		detections = dict()
		bboxes = np.vstack(bbox_result)
		# logger.info(f'the shape of bbox are:{bboxes.shape}')
		labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
		# logger.info(f'the shape of old labels are:{labels}')
		detections['boxes'] = bboxes[:,0:4].tolist()
		detections['scores'] = bboxes[:, -1].tolist()
		labels = np.concatenate(labels).tolist()
		# logger.info(f'the shape of new labels are:{labels}')
		# logger.info(f'the max of labels are:{max(labels)}')
		if(np.isnan(bboxes).any()):
			logger.info(f'filename:{dataset.filename(idx)}')

		# if np.isnan(labels) == 'True':
		# 	print(dataset.filename(idx))
		# if np.isnan(detections['scores']) == 'True':
		# 	print(dataset.filename(idx))

		remove_idx = []
		for j, obj in enumerate(labels):
			if str(obj) in coco2hico:
				labels[j] = coco2hico[str(obj)]
			else:
				remove_idx.append(j)
		detections['labels'] = labels
		# Remove detections of deprecated object classes
		remove_idx.sort(reverse=True)
		for j in remove_idx:
			detections['boxes'].pop(j)
			detections['scores'].pop(j)
			detections['labels'].pop(j)

		with open(os.path.join(
			cache_dir,
			dataset.filename(idx).replace('jpg', 'json')
		), 'w') as f:
			json.dump(detections, f)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Data preprocessing')
	parser.add_argument('--partition', type=str, default='train2015')
	parser.add_argument('--data-root', type=str, default='../')
	parser.add_argument('--cache-dir', type=str, default='./')
	parser.add_argument('--nms-thresh', type=float, default=0.5)
	parser.add_argument('--score-thresh', type=float, default=0.05)
	parser.add_argument('--num-detections-per-image', type=int, default=100)
	parser.add_argument('--ckpt-path', type=str, default='',
			help="Path to a checkpoint that contains the weights for a model")
	args = parser.parse_args()

	print(args)
	main(args)
