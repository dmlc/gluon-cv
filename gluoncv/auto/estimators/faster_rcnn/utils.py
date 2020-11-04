"""Utils for Faster RCNN estimator"""
import os

from mxnet import gluon

from ....data.batchify import FasterRCNNTrainBatchify, Tuple, Append
from ....data.sampler import SplitSortedBucketSampler
from ....data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from .... import data as gdata
from ....utils.metrics.coco_detection import COCODetectionMetric
from ....utils.metrics.voc_detection import VOC07MApMetric

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


def _get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    return lr_warmup_factor * (1 - alpha) + alpha


def _split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for _, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch


def _save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as log_file:
            log_file.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def _get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                    num_shards, args):
    """Get dataloader."""
    train_bfn = FasterRCNNTrainBatchify(net, num_shards)
    if hasattr(train_dataset, 'get_im_aspect_ratio'):
        im_aspect_ratio = train_dataset.get_im_aspect_ratio()
    else:
        im_aspect_ratio = [1.] * len(train_dataset)
    train_sampler = \
        SplitSortedBucketSampler(im_aspect_ratio, batch_size,
                                 num_parts=hvd.size() if args.horovod else 1,
                                 part_index=hvd.rank() if args.horovod else 0,
                                 shuffle=True)
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(
            train_transform(net.short, net.max_size, net, ashape=net.ashape,
                            multi_stage=args.faster_rcnn.use_fpn)),
        batch_sampler=train_sampler, batchify_fn=train_bfn, num_workers=args.num_workers)
    val_bfn = Tuple(*[Append() for _ in range(3)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    # validation use 1 sample per device
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size)), num_shards, False,
        batchify_fn=val_bfn, last_batch='keep', num_workers=args.num_workers)
    train_eval_loader = gluon.data.DataLoader(
        train_dataset.transform(val_transform(short, net.max_size)), num_shards, False,
        batchify_fn=val_bfn, last_batch='keep', num_workers=args.num_workers)
    return train_loader, val_loader, train_eval_loader


def _get_testloader(net, test_dataset, num_devices, config):
    """Get faster rcnn test dataloader."""
    if config.meta_arch == 'faster_rcnn':
        test_bfn = Tuple(*[Append() for _ in range(3)])
        short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
        # validation use 1 sample per device
        test_loader = gluon.data.DataLoader(
            test_dataset.transform(FasterRCNNDefaultValTransform(short, net.max_size)),
            num_devices,
            False,
            batchify_fn=test_bfn,
            last_batch='keep',
            num_workers=config.num_workers
        )
        return test_loader
    else:
        raise NotImplementedError('%s not implemented.' % config.meta_arch)


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
    elif dataset.lower() in ['clipart', 'comic', 'watercolor']:
        root = os.path.join('~', '.mxnet', 'datasets', dataset.lower())
        train_dataset = gdata.CustomVOCDetection(root=root, splits=[('', 'train')],
                                                 generate_classes=True)
        val_dataset = gdata.CustomVOCDetection(root=root, splits=[('', 'test')],
                                               generate_classes=True)
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset,
                                         os.path.join(args.logdir, args.save_prefix + '_eval'),
                                         cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.train.mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric
