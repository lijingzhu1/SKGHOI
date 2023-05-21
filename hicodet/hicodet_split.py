import os
import sys
import argparse
# sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs/hicodet')
# from hicodet import HICODet
from pocket.data import HICODetSubset,HICODet
from torch.utils.data import DataLoader, DistributedSampler
sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs')
from utils import custom_collate, CustomisedDLE, DataFactory
os.chdir('/users/PCS0256/lijing/spatially-conditioned-graphs/hicodet/detections')

def main(args):

    # trainset = DataFactory(
    # name=args.dataset, partition=args.partitions[0],
    # data_root=args.data_root,
    # detection_root=args.train_detection_dir,
    # flip=True
    # )

    # valset = DataFactory(
    #     name=args.dataset, partition=args.partitions[1],
    #     data_root=args.data_root,
    #     detection_root=args.val_detection_dir
    # )

    # train_loader = DataLoader(
    #     dataset=trainset,
    #     collate_fn=custom_collate, batch_size=args.batch_size,
    #     num_workers=args.num_workers, pin_memory=True
    # )

    # val_loader = DataLoader(
    #     dataset=valset,
    #     collate_fn=custom_collate, batch_size=args.batch_size,
    #     num_workers=args.num_workers, pin_memory=True
    # )

	dataset = HICODet(
        root=os.path.join(args.data_root,
            "hico_20160224_det/images/{}".format(args.partition)),
        anno_file=os.path.join(args.data_root,
            "instances_{}.json".format(args.partition))
    )

	dataset.split(0.5)
    # sepearter = HICODetSubset(train_loader,0.5)
    # sepearter()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Data spilting')
	parser.add_argument('--partition', type=str, default='train2015')
	parser.add_argument('--dataset', default='hicodet', type=str)
	parser.add_argument('--train-detection-dir', default='hicodet/detections/train2015', type=str)
	parser.add_argument('--val-detection-dir', default='hicodet/detections/test2015', type=str)
	parser.add_argument('--data-root', type=str, default='../')
	parser.add_argument('--batch-size', default=4, type=int,help="Batch size for each subprocess")
	parser.add_argument('--num-workers', default=1, type=int)
	parser.add_argument('--world-size', default=1, type=int,
                        help="Number of subprocesses/GPUs to use")

	args = parser.parse_args()

	print(args)
	main(args)



