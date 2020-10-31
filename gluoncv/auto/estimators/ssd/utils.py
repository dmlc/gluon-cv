"""Utils for auto SSD estimator"""
import os

import mxnet as mx
from mxnet import gluon
from mxnet import autograd


from ....data.batchify import Tuple, Stack, Pad
from ....data.transforms.presets.ssd import SSDDefaultTrainTransform
from ....data.transforms.presets.ssd import SSDDefaultValTransform
from ....data.transforms.presets.ssd import SSDDALIPipeline
from .... import data as gdata
from ....utils.metrics.voc_detection import VOC07MApMetric
from ....utils.metrics.coco_detection import COCODetectionMetric

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

try:
    from nvidia.dali.plugin.mxnet import DALIGenericIterator
    dali_found = True
except ImportError:
    dali_found = False


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
        train_dataset = gdata.COCODetection(splits=['instances_train2017'])
        val_dataset = gdata.COCODetection(splits=['instances_val2017'], skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, os.path.join(args.logdir, args.save_prefix + '_eval'), cleanup=True,
            data_shape=(args.ssd.data_shape, args.ssd.data_shape))
        # coco validation is slow, consider increase the validation interval
        if args.validation.val_interval == 1:
            args.validation.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric

def _get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    anchors = anchors.as_in_context(mx.cpu())
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    train_eval_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader, train_eval_loader

def _get_dali_dataset(dataset_name, devices, args):
    if dataset_name.lower() == "coco":
        # training
        expanded_file_root = os.path.expanduser(args.dataset_root)
        coco_root = os.path.join(expanded_file_root,
                                 'coco',
                                 'train2017')
        coco_annotations = os.path.join(expanded_file_root,
                                        'coco',
                                        'annotations',
                                        'instances_train2017.json')
        if args.horovod:
            train_dataset = [gdata.COCODetectionDALI(num_shards=hvd.size(), shard_id=hvd.rank(), file_root=coco_root,
                                                     annotations_file=coco_annotations, device_id=hvd.local_rank())]
        else:
            train_dataset = [gdata.COCODetectionDALI(num_shards=len(devices), shard_id=i, file_root=coco_root,
                                                     annotations_file=coco_annotations, device_id=i) \
                                                     for i, _ in enumerate(devices)]

        # validation
        if not args.horovod or hvd.rank() == 0:
            val_dataset = gdata.COCODetection(root=os.path.join(args.dataset_root + '/coco'),
                                              splits='instances_val2017',
                                              skip_empty=False)
            val_metric = COCODetectionMetric(
                val_dataset, os.path.join(args.logdir, args.save_prefix + '_eval'), cleanup=True,
                data_shape=(args.ssd.data_shape, args.ssd.data_shape))
        else:
            val_dataset = None
            val_metric = None
    else:
        raise NotImplementedError('Dataset: {} not implemented with DALI.'.format(dataset_name))

    return train_dataset, val_dataset, val_metric

def _get_dali_dataloader(net, train_dataset, val_dataset, data_shape, global_batch_size,
                         num_workers, devices, ctx, horovod):
    width, height = data_shape, data_shape
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx=ctx))
    anchors = anchors.as_in_context(mx.cpu())

    if horovod:
        batch_size = global_batch_size // hvd.size()
        pipelines = [SSDDALIPipeline(device_id=hvd.local_rank(), batch_size=batch_size,
                                     data_shape=data_shape, anchors=anchors,
                                     num_workers=num_workers, dataset_reader=train_dataset[0])]
    else:
        num_devices = len(devices)
        batch_size = global_batch_size // num_devices
        pipelines = [SSDDALIPipeline(device_id=device_id, batch_size=batch_size,
                                     data_shape=data_shape, anchors=anchors,
                                     num_workers=num_workers,
                                     dataset_reader=train_dataset[i]) for i, device_id in enumerate(devices)]

    epoch_size = train_dataset[0].size()
    if horovod:
        epoch_size //= hvd.size()
    train_loader = DALIGenericIterator(pipelines, [('data', DALIGenericIterator.DATA_TAG),
                                                   ('bboxes', DALIGenericIterator.LABEL_TAG),
                                                   ('label', DALIGenericIterator.LABEL_TAG)],
                                       epoch_size, auto_reset=True)

    # validation
    if not horovod or hvd.rank() == 0:
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(SSDDefaultValTransform(width, height)),
            global_batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    else:
        val_loader = None

    return train_loader, val_loader

def _save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_{:04d}_{:.4f}best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as log_file:
            log_file.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))
