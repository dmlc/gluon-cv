"""Utils for auto Mask-RCNN estimator"""
# pylint: disable=consider-using-enumerate
import os
import logging

import mxnet as mx
from mxnet import gluon

from .... import data as gdata
from ....data import batchify
from ....data.sampler import SplitSortedBucketSampler
from ....utils.metrics.coco_instance import COCOInstanceMetric

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

try:
    from mpi4py import MPI
except ImportError:
    logging.info('mpi4py is not installed. Use "pip install --no-cache mpi4py" to install')
    MPI = None


def _get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        train_dataset = gdata.COCOInstance(splits='instances_train2017')
        val_dataset = gdata.COCOInstance(splits='instances_val2017', skip_empty=False)
        starting_id = 0
        if args.horovod and MPI:
            length = len(val_dataset)
            shard_len = length // hvd.size()
            rest = length % hvd.size()
            # Compute the start index for this partition
            starting_id = shard_len * hvd.rank() + min(hvd.rank(), rest)
        val_metric = COCOInstanceMetric(val_dataset, os.path.join(args.logdir, args.save_prefix + '_eval'),
                                        use_ext=args.use_ext, starting_id=starting_id)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.horovod and MPI:
        val_dataset = val_dataset.shard(hvd.size(), hvd.rank())
    return train_dataset, val_dataset, val_metric


def _get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                    num_shards_per_process, args):
    """Get dataloader."""
    train_bfn = batchify.MaskRCNNTrainBatchify(net, num_shards_per_process)
    train_sampler = \
        SplitSortedBucketSampler(train_dataset.get_im_aspect_ratio(),
                                 batch_size,
                                 num_parts=hvd.size() if args.horovod else 1,
                                 part_index=hvd.rank() if args.horovod else 0,
                                 shuffle=True)
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(train_transform(net.short, net.max_size,
                                                net, ashape=net.ashape, multi_stage=args.mask_rcnn.use_fpn)),
        batch_sampler=train_sampler, batchify_fn=train_bfn, num_workers=args.num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    # validation use 1 sample per device
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size)), num_shards_per_process, False,
        batchify_fn=val_bfn, last_batch='keep', num_workers=args.num_workers)
    return train_loader, val_loader


def _save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as log_file:
            log_file.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def _stage_data(i, data, ctx_list, pinned_data_stage):
    def _get_chunk(data, storage):
        s = storage.reshape(shape=(storage.size,))
        s = s[:data.size]
        s = s.reshape(shape=data.shape)
        data.copyto(s)
        return s

    if ctx_list[0].device_type == "cpu":
        return data
    if i not in pinned_data_stage:
        pinned_data_stage[i] = [d.as_in_context(mx.cpu_pinned()) for d in data]
        return pinned_data_stage[i]

    storage = pinned_data_stage[i]

    for j in range(len(storage)):
        if data[j].size > storage[j].size:
            storage[j] = data[j].as_in_context(mx.cpu_pinned())

    return [_get_chunk(d, s) for d, s in zip(data, storage)]


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


def _get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    return lr_warmup_factor * (1 - alpha) + alpha
