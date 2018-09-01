import argparse, os

import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.nn import Block, HybridBlock
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model

# CLI
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
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--params-file', type=str,
                    help='local parameter file to load, instead of pre-trained weight.')
parser.add_argument('--dtype', type=str,
                    help='training data type')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
opt = parser.parse_args()

batch_size = opt.batch_size
classes = 1000

num_gpus = opt.num_gpus
batch_size *= num_gpus
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

model_name = opt.model
pretrained = True if not opt.params_file else False

kwargs = {'ctx': ctx, 'pretrained': pretrained, 'classes': classes}
if model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

net = get_model(model_name, **kwargs)
net.cast(opt.dtype)
if opt.params_file:
    net.load_params(opt.params_file, ctx=ctx)
net.hybridize()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.Resize(256, keep_ratio=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def test(ctx, val_data, mode='image'):
    acc_top1.reset()
    acc_top5.reset()
    if not opt.rec_dir:
        num_batch = len(val_data)
    for i, batch in enumerate(val_data):
        if mode == 'image':
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        else:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        if not opt.rec_dir:
            print('%d / %d : %.8f, %.8f'%(i, num_batch, 1-top1, 1-top5))
        else:
            print('%d : %.8f, %.8f'%(i, 1-top1, 1-top5))

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

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
        preprocess_threads  = 30,
        batch_size          = batch_size,

        resize              = 256,
        data_shape          = (3, 224, 224),
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        std_r               = 58.393,
        std_g               = 57.12,
        std_b               = 57.375
    )

if not opt.rec_dir:
    err_top1_val, err_top5_val = test(ctx, val_data, 'image')
else:
    err_top1_val, err_top5_val = test(ctx, val_data, 'rec')
print(err_top1_val, err_top5_val)

params_count = 0
kwargs2 = {'ctx': mx.cpu(), 'pretrained': False, 'classes': classes}
net2 = get_model(model_name, **kwargs2)
net2.initialize()
p = net2(mx.nd.zeros((1, 3, 224, 224)))
for k, v in net2.collect_params().items():
    params_count += v.data().size

print(params_count)
