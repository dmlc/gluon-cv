import argparse, time, logging, os, math, sys

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
from gluoncv.nn.block import DSNT
from gluoncv.data.transforms.pose import transform_preds, get_final_preds, flip_heatmap
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform, SimplePoseDefaultValTransform
from gluoncv.utils.metrics.coco_keypoints import COCOKeyPointsMetric

# CLI
parser = argparse.ArgumentParser(description='Validate a model for pose estimation.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/coco',
                    help='training and validation pictures to use.')
parser.add_argument('--num-joints', type=int, required=True,
                    help='Number of joints to detect')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--model-prefix', type=str, required=False,
                    help='load static model as hybridblock.')
parser.add_argument('--deploy', action='store_true',
                    help='whether load static model for deployment')
parser.add_argument('--quantized', action='store_true', 
                    help='whether to use int8 pretrained  model')
parser.add_argument('--num-iterations', type=int, default=100,
                    help='number of benchmarking iterations.')
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
parser.add_argument('--dsnt', action='store_true',
                    help='Whether to use dsnt to approximate coordinates.')
parser.add_argument('--mean', type=str, default='0.485,0.456,0.406',
                    help='mean vector for normalization')
parser.add_argument('--std', type=str, default='0.229,0.224,0.225',
                    help='std vector for normalization')
parser.add_argument('--score-threshold', type=float, default=0,
                    help='threshold value for predicted score.')
# dummy benchmark
parser.add_argument('--benchmark', action='store_true',
                    help='whether to use dummy data for benchmarking performance.')
# calibration
parser.add_argument('--calibration', action='store_true',
                    help='quantize model')
parser.add_argument('--num-calib-batches', type=int, default=5,
                    help='number of batches for calibration')
parser.add_argument('--quantized-dtype', type=str, default='auto', 
                    choices=['auto', 'int8', 'uint8'],
                    help='quantization destination data type for input data')
parser.add_argument('--calib-mode', type=str, default='naive',
                    help='calibration mode used for generating calibration table for the quantized symbol; supports'
                        ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                        ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                        ' in general.'
                        ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                        ' quantization. In general, the inference accuracy worsens with more examples used in'
                        ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                        ' inference results.'
                        ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                        ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                        ' kinds of quantized models if the calibration dataset is representative enough of the'
                        ' inference dataset.')
opt = parser.parse_args()

logging.basicConfig()
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
logging.info(opt)

batch_size = opt.batch_size
num_joints = 17

num_gpus = opt.num_gpus
context = [mx.cpu()]
if num_gpus > 0:
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)]

num_workers = opt.num_workers

def get_data_loader(data_dir, batch_size, num_workers, input_size):

    def val_batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                          batch_axis=0, even_split=False)
        scale = batch[1]
        center = batch[2]
        score = batch[3]
        imgid = batch[4]
        return data, scale, center, score, imgid

    val_dataset = mscoco.keypoints.COCOKeyPoints(data_dir, splits=('person_keypoints_val2017'))

    meanvec = [float(i) for i in opt.mean.split(',')]
    stdvec = [float(i) for i in opt.std.split(',')]
    transform_val = SimplePoseDefaultValTransform(num_joints=val_dataset.num_joints,
                                                  joint_pairs=val_dataset.joint_pairs,
                                                  image_size=input_size,
                                                  mean=meanvec,
                                                  std=stdvec)
    val_data = gluon.data.DataLoader(
        val_dataset.transform(transform_val),
        batch_size=batch_size, shuffle=False, last_batch='keep',
        num_workers=num_workers)

    return val_dataset, val_data, val_batch_fn

input_size = [int(i) for i in opt.input_size.split(',')]

if opt.calibration or not opt.benchmark:
    val_dataset, val_data, val_batch_fn = get_data_loader(opt.data_dir, batch_size,
                                                          num_workers, input_size)
    val_metric = COCOKeyPointsMetric(val_dataset, 'coco_keypoints',
                                     data_shape=tuple(input_size),
                                     in_vis_thresh=opt.score_threshold)

use_pretrained = True if not opt.params_file else False
model_name = opt.model if not opt.quantized else '_'.join([opt.model, 'int8'])

if not opt.deploy:
    net = get_model(model_name, ctx=context, num_joints=num_joints, pretrained=use_pretrained)
    if not use_pretrained:
        net.load_parameters(opt.params_file, ctx=context)
    if opt.quantized:
        net.hybridize(static_alloc=True, static_shape=True)
    else:
        net.hybridize()
else:
    model_name = 'deploy'
    net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(opt.model_prefix),
              ['data'], '{}-0000.params'.format(opt.model_prefix))
    net.hybridize(static_alloc=True, static_shape=True)

print("Inference on model {} started!".format(model_name))

# calibration on FP32 model
def calibration(net, val_data, opt, ctx, logger):
    exclude_sym_layer = []
    exclude_match_layer = []
    if num_gpus > 0:
        raise ValueError('currently only supports CPU with MKL-DNN backend')
    net = quantize_net(net, calib_data=val_data, quantized_dtype=opt.quantized_dtype, calib_mode=opt.calib_mode, 
                       exclude_layers=exclude_sym_layer, num_calib_examples=opt.batch_size * opt.num_calib_batches,
                       exclude_layers_match=exclude_match_layer, ctx=ctx, logger=logger)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dst_dir = os.path.join(dir_path, 'model')
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    prefix = os.path.join(dst_dir, opt.model + '-quantized-' + opt.calib_mode)
    logger.info('Saving quantized model at %s' % dst_dir)
    net.export(prefix, epoch=0)


if opt.dsnt:
    heatmap_size = [int(i/4) for i in input_size]
    net_dsnt = DSNT(size=heatmap_size[::-1])
    net_dsnt.initialize(ctx=context)
    net_dsnt.hybridize()

def validate(val_data, val_dataset, net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    val_metric.reset()

    from tqdm import tqdm
    for batch in tqdm(val_data):
        data, scale, center, score, imgid = val_batch_fn(batch, ctx)

        outputs = [net(X) for X in data]
        if opt.flip_test:
            data_flip = [nd.flip(X, axis=3) for X in data]
            outputs_flip = [net(X) for X in data_flip]
            outputs_flipback = [flip_heatmap(o, val_dataset.joint_pairs, shift=True) for o in outputs_flip]
            outputs = [(o + o_flip)/2 for o, o_flip in zip(outputs, outputs_flipback)]

        if opt.dsnt:
            outputs = [net_dsnt(X)[0] for X in outputs]

        if len(outputs) > 1:
            outputs_stack = nd.concat(*[o.as_in_context(mx.cpu()) for o in outputs], dim=0)
        else:
            outputs_stack = outputs[0].as_in_context(mx.cpu())

        if opt.dsnt:
            preds = (outputs_stack - 0.5) * scale.expand_dims(axis=1) + center.expand_dims(axis=1)
            maxvals = nd.ones(preds.shape[0:2]+(1, ))
        else:
            preds, maxvals = get_final_preds(outputs_stack, center.asnumpy(), scale.asnumpy())
        val_metric.update(preds, maxvals, score, imgid)

    metric_name, metric_score = val_metric.get()
    print("Inference Completed! %s = %.4f" % (metric_name, metric_score))
    return


def benchmarking(net, opt, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    bs = opt.batch_size
    num_iterations = opt.num_iterations
    input_shape = (bs, 3,) + tuple(input_size)
    size = num_iterations * bs
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx[0], dtype='float32')
    dry_run = 5

    from tqdm import tqdm
    with tqdm(total=size + dry_run * bs) as pbar:
        for n in range(dry_run + num_iterations):
            if n == dry_run:
                tic = time.time()
            output = net(data)
            output.wait_to_read()
            pbar.update(bs)
    speed = size / (time.time() - tic)
    print('With batch size %d , %d batches, throughput is %f imgs/sec' % (bs, num_iterations, speed))


if __name__ == '__main__':
    if opt.calibration:
        calibration(net, val_data, opt, context[0], logger)
        sys.exit()

    if opt.benchmark:
        print("---------- Benchmarking on %s model -------------" % model_name)
        benchmarking(net, opt, context)
    else:
        validate(val_data, val_dataset, net, context)
