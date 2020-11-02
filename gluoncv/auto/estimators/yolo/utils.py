"""Utils for auto YOLO estimator"""
import os

from mxnet import gluon

from ....data import MixupDetection
from ....data.batchify import Tuple, Stack, Pad
from ....data.dataloader import RandomTransformDataLoader
from ....data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from ....data.transforms.presets.yolo import YOLO3DefaultValTransform
from .... import data as gdata
from ....utils.metrics.voc_detection import VOC07MApMetric
from ....utils.metrics.coco_detection import COCODetectionMetric


def _get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'voc_tiny':
        # need to download the dataset and specify the path to store the dataset in
        # root = os.path.expanduser('~/.mxnet/datasets/')
        # filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip', path=root)
        # filename = ag.unzip(filename_zip, root=root)
        # data_root = os.path.join(root, filename)
        train_dataset = gdata.CustomVOCDetectionBase(classes=('motorbike',), root=args.dataset_root + 'tiny_motorbike',
                                                     splits=[('', 'trainval')])
        val_dataset = gdata.CustomVOCDetectionBase(classes=('motorbike',), root=args.dataset_root + 'tiny_motorbike',
                                                   splits=[('', 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, os.path.join(args.logdir, args.save_prefix + '_eval'), cleanup=True,
            data_shape=(args.yolo3.data_shape, args.yolo3.data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.train.num_samples < 0:
        args.train.num_samples = len(train_dataset)
    if args.train.mixup:
        train_dataset = MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric

def _get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # stack image, all targets generated
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
    if args.yolo3.no_random_shape:
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.train.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.train.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    train_eval_loader = gluon.data.DataLoader(
        train_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader, train_eval_loader

def _save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_{:04d}_{:.4f}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as log_file:
            log_file.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))
