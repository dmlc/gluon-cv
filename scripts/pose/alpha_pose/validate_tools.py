import sys

from tqdm import tqdm

import mxnet as mx
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import mscoco
from gluoncv.data.transforms.pose import (flip_heatmap,
                                          heatmap_to_coord_alpha_pose)
from gluoncv.data.transforms.presets.alpha_pose import \
    AlphaPoseDefaultValTransform
from gluoncv.utils.metrics.coco_keypoints import COCOKeyPointsMetric
from mxnet import gluon, nd


class NullWriter(object):
    def write(self, arg):
        pass


def get_dataset(dataset):
    if dataset == 'coco':
        val_dataset = mscoco.keypoints.COCOKeyPoints(splits=('person_keypoints_val2017'), skip_empty=False)
    else:
        raise NotImplementedError("Dataset: {} not supported.".format(dataset))
    return val_dataset


def val_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                      batch_axis=0, even_split=False)
    return tuple([data] + batch[1:])


def get_val_data_loader(dataset, batch_size, num_workers, input_size, opt):
    val_dataset = get_dataset(dataset)

    transform_val = AlphaPoseDefaultValTransform(num_joints=val_dataset.num_joints,
                                                 joint_pairs=val_dataset.joint_pairs,
                                                 image_size=input_size)
    val_data = gluon.data.DataLoader(
        val_dataset.transform(transform_val),
        batch_size=batch_size, shuffle=False, last_batch='keep',
        num_workers=num_workers)

    return val_dataset, val_data


def validate(val_data, val_dataset, net, ctx, opt):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    val_metric = COCOKeyPointsMetric(val_dataset, 'coco_keypoints',
                                     in_vis_thresh=0)

    for batch in tqdm(val_data, dynamic_ncols=True):
        # data, scale, center, score, imgid = val_batch_fn(batch, ctx)
        data, scale_box, score, imgid = val_batch_fn(batch, ctx)

        outputs = [net(X) for X in data]
        if opt.flip_test:
            data_flip = [nd.flip(X, axis=3) for X in data]
            outputs_flip = [net(X) for X in data_flip]
            outputs_flipback = [flip_heatmap(o, val_dataset.joint_pairs, shift=True) for o in outputs_flip]
            outputs = [(o + o_flip) / 2 for o, o_flip in zip(outputs, outputs_flipback)]

        if len(outputs) > 1:
            outputs_stack = nd.concat(*[o.as_in_context(mx.cpu()) for o in outputs], dim=0)
        else:
            outputs_stack = outputs[0].as_in_context(mx.cpu())

        # preds, maxvals = get_final_preds(outputs_stack, center.asnumpy(), scale.asnumpy())
        preds, maxvals = heatmap_to_coord_alpha_pose(outputs_stack, scale_box)
        val_metric.update(preds, maxvals, score, imgid)

    nullwriter = NullWriter()
    oldstdout = sys.stdout
    sys.stdout = nullwriter
    try:
        res = val_metric.get()
    finally:
        sys.stdout = oldstdout
    return res
