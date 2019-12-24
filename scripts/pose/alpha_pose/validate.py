import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
from gluoncv.data.transforms.pose import transform_preds, get_final_preds, flip_heatmap, heatmap_to_coord_alpha_pose
from gluoncv.data.transforms.presets.alpha_pose import AlphaPoseDefaultValTransform
from gluoncv.utils.metrics.coco_keypoints import COCOKeyPointsMetric

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--dataset', type=str, default='coco',
                    help='training and validation pictures to use.')
parser.add_argument('--num-joints', type=int, required=True,
                    help='Number of joints to detect')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--input-size', type=str, default='256,192',
                    help='size of the input image size. default is 256,192')
parser.add_argument('--params-file', type=str,
                    help='local parameters to load.')
parser.add_argument('--flip-test', action='store_true',
                    help='Whether to flip test input to ensemble results.')
parser.add_argument('--score-threshold', type=float, default=0,
                    help='threshold value for predicted score.')
opt = parser.parse_args()

batch_size = opt.batch_size
num_joints = 17

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

def get_dataset(dataset):
    if dataset == 'coco':
        val_dataset = mscoco.keypoints.COCOKeyPoints(splits=('person_keypoints_val2017'), skip_empty=False)
    else:
        raise NotImplementedError("Dataset: {} not supported.".format(dataset))
    return val_dataset

def get_data_loader(dataset, batch_size, num_workers, input_size):

    def val_batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                          batch_axis=0, even_split=False)
        return tuple([data] + batch[1:])

    val_dataset = get_dataset(dataset)

    transform_val = AlphaPoseDefaultValTransform(num_joints=val_dataset.num_joints,
                                                 joint_pairs=val_dataset.joint_pairs,
                                                 image_size=input_size)
    val_data = gluon.data.DataLoader(
        val_dataset.transform(transform_val),
        batch_size=batch_size, shuffle=False, last_batch='keep',
        num_workers=num_workers)

    return val_dataset, val_data, val_batch_fn

input_size = [int(i) for i in opt.input_size.split(',')]
val_dataset, val_data, val_batch_fn = get_data_loader(opt.dataset, batch_size,
                                                      num_workers, input_size)
val_metric = COCOKeyPointsMetric(val_dataset, 'coco_keypoints',
                                 data_shape=tuple(input_size),
                                 in_vis_thresh=opt.score_threshold)

use_pretrained = True if not opt.params_file else False
model_name = '_'.join((opt.model, opt.dataset))
kwargs = {'ctx': context,
          'pretrained': use_pretrained,
          'num_gpus': num_gpus}
net = get_model(model_name, **kwargs)
if not use_pretrained:
    net.load_parameters(opt.params_file, ctx=context)
net.hybridize()

def validate(val_data, val_dataset, net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    val_metric.reset()

    from tqdm import tqdm
    for batch in tqdm(val_data):
        # data, scale, center, score, imgid = val_batch_fn(batch, ctx)
        data, scale_box, score, imgid = val_batch_fn(batch, ctx)

        outputs = [net(X) for X in data]
        if opt.flip_test:
            data_flip = [nd.flip(X, axis=3) for X in data]
            outputs_flip = [net(X) for X in data_flip]
            outputs_flipback = [flip_heatmap(o, val_dataset.joint_pairs, shift=True) for o in outputs_flip]
            outputs = [(o + o_flip)/2 for o, o_flip in zip(outputs, outputs_flipback)]

        if len(outputs) > 1:
            outputs_stack = nd.concat(*[o.as_in_context(mx.cpu()) for o in outputs], dim=0)
        else:
            outputs_stack = outputs[0].as_in_context(mx.cpu())

        # preds, maxvals = get_final_preds(outputs_stack, center.asnumpy(), scale.asnumpy())
        preds, maxvals = heatmap_to_coord_alpha_pose(outputs_stack, scale_box)
        # print(preds, maxvals, scale_box)
        # print(preds, maxvals)
        # raise
        val_metric.update(preds, maxvals, score, imgid)

    res = val_metric.get()
    return

if __name__ == '__main__':
    validate(val_data, val_dataset, net, context)
