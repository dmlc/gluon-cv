import argparse, os, math, time, sys

import mxnet as mx
import logging
from mxnet import gluon, nd, image
from mxnet.gluon.nn import Block, HybridBlock
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='Imagenet directory for validation.')
    parser.add_argument('--rec-dir', type=str, default='',
                        help='recio directory for validation.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--model', type=str, default='model', required=False,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--deploy', action='store_true',
                        help='whether load static model for deployment')
    parser.add_argument('--model-prefix', type=str, required=False,
                        help='load static model as hybridblock.')
    parser.add_argument('--quantized', action='store_true',
                        help='use int8 pretrained model')
    parser.add_argument('--input-size', type=int, default=224,
                        help='input shape of the image, default is 224.')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='run specified number of batches for inference')
    parser.add_argument('--benchmark', action='store_true',
                        help='use synthetic data to evalute benchmark')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='The ratio for crop and input size, for validation dataset only')
    parser.add_argument('--params-file', type=str,
                        help='local parameter file to load, instead of pre-trained weight.')
    parser.add_argument('--dtype', type=str,
                        help='training data type')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
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
    return opt

def benchmark(network, ctx, batch_size=64, image_size=224, num_iter=100, datatype='float32'):
    input_shape = (batch_size, 3) + (image_size, image_size)
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx, dtype=datatype)
    dryrun = 5
    for i in range(num_iter+dryrun):
        if i == dryrun:
            tic = time.time()
        output = network(data)
        output.asnumpy()
    toc = time.time() - tic
    return toc

def test(network, ctx, val_data, mode='image'):
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    if not opt.rec_dir:
        num_batch = len(val_data)
    num = 0
    start = time.time()
    for i, batch in enumerate(val_data):
        if mode == 'image':
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        else:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = [network(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        if not opt.rec_dir:
            print('%d / %d : %.8f, %.8f'%(i, num_batch, 1-top1, 1-top5))
        else:
            print('%d : %.8f, %.8f'%(i, 1-top1, 1-top5))
        num += batch_size
    end = time.time()
    speed = num / (end - start)
    print('Throughput is %f img/sec.'% speed)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

if __name__ == '__main__':
    opt = parse_args()
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.info(opt)

    batch_size = opt.batch_size
    classes = 1000

    num_gpus = opt.num_gpus
    if num_gpus > 0:
        batch_size *= num_gpus
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    input_size = opt.input_size
    model_name = opt.model
    if opt.quantized:
        model_name = '_'.join((model_name, 'int8'))
    pretrained = True if not opt.params_file else False

    kwargs = {'ctx': ctx, 'pretrained': pretrained, 'classes': classes}
    if model_name.startswith('resnext'):
        kwargs['use_se'] = opt.use_se

    if opt.deploy:
        model_name = 'deploy'
        net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(opt.model_prefix),
              ['data'], '{}-0000.params'.format(opt.model_prefix))
        net.hybridize(static_alloc=True, static_shape=True)
    else:
        net = get_model(model_name, **kwargs)
        net.cast(opt.dtype)
        if opt.params_file:
            net.load_parameters(opt.params_file, ctx=ctx)
        if opt.quantized:
            net.hybridize(static_alloc=True, static_shape=True)
        else:
            net.hybridize()

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    """
    Aligning with TF implementation, the default crop-input
    ratio set as 0.875; Set the crop as ceil(input-size/ratio)
    """
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size/crop_ratio))

    transform_test = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    if not opt.benchmark:
        if not opt.rec_dir:
            val_data = gluon.data.DataLoader(
                imagenet.classification.ImageNet(opt.data_dir, train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            imgrec = os.path.join(opt.rec_dir, 'val.rec')
            imgidx = os.path.join(opt.rec_dir, 'val.idx')
            val_data = mx.io.ImageRecordIter(
                path_imgrec         = imgrec,
                path_imgidx         = imgidx,
                preprocess_threads  = num_workers,
                batch_size          = batch_size,

                resize              = resize,
                data_shape          = (3, input_size, input_size),
                mean_r              = 123.68,
                mean_g              = 116.779,
                mean_b              = 103.939,
                std_r               = 58.393,
                std_g               = 57.12,
                std_b               = 57.375
            )

    if opt.calibration and not opt.quantized:
        exclude_layers = []
        exclude_layers_match = ['flatten']
        logger.info('quantize net with batch size = %d', batch_size)
        if num_gpus > 0:
            raise ValueError('currently only supports CPU with MKL-DNN backend')
        net = quantize_net(
            net, quantized_dtype='auto', exclude_layers=exclude_layers,
            exclude_layers_match=exclude_layers_match, calib_data=val_data,
            calib_mode=opt.calib_mode, num_calib_examples=batch_size * opt.num_calib_batches, ctx=ctx[0],
            logger=logger)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dst_dir = os.path.join(dir_path, 'model')
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        prefix = os.path.join(dst_dir, model_name +
                              '-quantized-' + opt.calib_mode)
        logger.info('Saving quantized model at %s' % dir_path)
        net.export(prefix, epoch=0)
        net.hybridize(static_alloc=True, static_shape=True)
        sys.exit()

    if opt.benchmark:
        print('-----benchmark mode for model %s-----'%model_name)
        time_cost = benchmark(network=net, ctx=ctx[0], image_size=opt.input_size, batch_size=opt.batch_size,
            num_iter=opt.num_batches, datatype='float32')
        fps = (opt.batch_size*opt.num_batches)/time_cost
        print('With batch size %s, %s batches, inference performance is %.2f img/sec' % (opt.batch_size, opt.num_batches, fps))
        sys.exit()

    if not opt.rec_dir:
        err_top1_val, err_top5_val = test(net, ctx, val_data, 'image')
    else:
        err_top1_val, err_top5_val = test(net, ctx, val_data, 'rec')
    print(err_top1_val, err_top5_val)

    params_count = 0
    kwargs2 = {'ctx': mx.cpu(), 'pretrained': False, 'classes': classes}
    net2 = get_model(model_name, **kwargs2)
    net2.initialize()
    p = net2(mx.nd.zeros((1, 3, input_size, input_size)))
    for k, v in net2.collect_params().items():
        params_count += v.data().size

    print(params_count)
