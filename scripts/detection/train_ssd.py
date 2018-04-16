"""
Train SSD on Pascal VOC dataset
===============================

This article walk you through the components GluonVision provided to you
that are very useful to start an object detection project. By going
through this tutorial, we show how stacking the existing modules can
produce a SOTA Single Shot Multibox Detection [1] model.

**Feel free to skip this tutorial because the training script is
self-complete and only requires a single command line to launch.**

Dataset
-------

We hope you already read this `article <http://gluon-vision.mxnet.io.s3-website-us-west-2.amazonaws.com/examples/datasets/setup_pascal_voc.html>`__ so Pascal VOC dataset is
well sitting on your disk. If so we are ready to load some training and
validation images.

.. code:: python

    from gluonvision.data import VOCDetection
    # typically we use 2007+2012 trainval splits as training data
    train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
    # use 2007 test as validation
    val_dataset = VOCDetection(splits=[(2007, 'test')])

    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))


.. parsed-literal::

    Training images: 16551
    Validation images: 4952


Data Transform
--------------

We can read a image and label pair from training dataset:

.. code:: python

    train_image, train_label = train_dataset[0]
    bboxes = train_label[:, :4]
    cids = train_label[:, 4:5]
    print('image:', train_image.shape)
    print('bboxes:', bboxes.shape, 'class ids:', cids.shape)


.. parsed-literal::

    image: (375, 500, 3)
    bboxes: (5, 4) class ids: (5, 1)


We could illustrate the image, together with the bounding box labels.

.. code:: python

    from matplotlib import pyplot as plt
    %matplotlib inline
    from gluonvision.utils import viz

    ax = viz.plot_bbox(train_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
    plt.show()



.. image:: https://github.com/zhreshold/gluonvision-tutorials/blob/master/detection/ssd_train_voc/output_6_0.png?raw=true


At this point, validation images are quite similar.

.. code:: python

    val_image, val_label = val_dataset[0]
    bboxes = val_label[:, :4]
    cids = val_label[:, 4:5]
    ax = viz.plot_bbox(val_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
    plt.show()



.. image:: https://github.com/zhreshold/gluonvision-tutorials/blob/master/detection/ssd_train_voc/output_8_0.png?raw=true


For SSD networks, it is critical to apply data augmentation (see
explanations in paper [1]). We provide tons of image and bounding box
transform functions to supply that. It is very convenient to use as
well.

.. code:: python

    from gluonvision.data.transforms import presets
    from gluonvision import utils
    from mxnet import nd

    width, height = 512, 512  # suppose we use 512 as base training size
    train_transform = presets.ssd.SSDDefaultTrainTransform(width, height)
    val_transform = presets.ssd.SSDDefaultValTransform(width, height)

    utils.random.seed(233)  # fix seed in this tutorial

    # apply transforms to train image
    train_image2, train_label2 = train_transform(train_image, train_label)
    print('tensor shape:', train_image2.shape)


.. parsed-literal::

    tensor shape: (3, 512, 512)


Images directly from tensor is distorted because they no longer sit in
(0, 255) range. Let's convert it back so we can see it clearly.

.. code:: python

    train_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    train_image2 = (train_image2 * 255).clip(0, 255)
    ax = viz.plot_bbox(train_image2.asnumpy(), train_label2[:, :4],
                       labels=train_label2[:, 4:5], class_names=train_dataset.classes)
    plt.show()

    # apply transforms to validation image
    val_image2, val_label2 = val_transform(val_image, val_label)
    val_image2 = val_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    val_image2 = (val_image2 * 255).clip(0, 255)
    ax = viz.plot_bbox(val_image2.clip(0, 255).asnumpy(), val_label2[:, :4],
                       labels=val_label2[:, 4:5], class_names=train_dataset.classes)
    plt.show()



.. image:: https://github.com/zhreshold/gluonvision-tutorials/blob/master/detection/ssd_train_voc/output_12_0.png?raw=true



.. image:: https://github.com/zhreshold/gluonvision-tutorials/blob/master/detection/ssd_train_voc/output_12_1.png?raw=true


Transforms used in training include random expanding, random cropping,
color distortion, random flipping, etc. In comparison, validation
transforms are conservative, where only resizing and color normalization
is used.

DataLoader
----------

We want iterate through the entire dataset many times during training.
Keep in mind that raw images have to be transformed into tensors(mxnet
use BCHW format) before they are fed into neural networks. Besides, to
be able to run in mini-batches, images must be resized to same shape.

A handy DataLoader would be very convenient for us to apply different
transforms and aggregate data into mini-batches.

Because number of objects varys a lot in different images, we have
fluctuating label sizes. As a result, we need to pad those labels to the
same size. In response, we have DetectionDataLoader ready for you which
handles it automatically.

.. code:: python

    from gluonvision.data import DetectionDataLoader

    batch_size = 4  # for tutorial, we use smaller batch-size
    num_workers = 4  # multi processing worker to accelerate data processing

    train_loader = DetectionDataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                                       last_batch='rollover', num_workers=num_workers)
    val_loader = DetectionDataLoader(val_dataset.transform(val_transform), batch_size, shuffle=False,
                                     last_batch='keep', num_workers=num_workers)

    for ib, batch in enumerate(train_loader):
        if ib > 5:
            break
        print('data:', batch[0].shape, 'label:', batch[1].shape)


.. parsed-literal::

    data: (4, 3, 512, 512) label: (4, 1, 6)
    data: (4, 3, 512, 512) label: (4, 1, 6)
    data: (4, 3, 512, 512) label: (4, 1, 6)
    data: (4, 3, 512, 512) label: (4, 6, 6)
    data: (4, 3, 512, 512) label: (4, 2, 6)
    data: (4, 3, 512, 512) label: (4, 5, 6)


SSD network
-----------

SSD network is a composite Gluon HybridBlock(which means it can be
exported to symbol to run in C++, Scala and other language bindings, but
we will cover it future tutorials). In terms of structure, SSD networks
are composed of feature extraction base network, anchor generators,
class predictors and bounding box offsets predictors. If you have read
our introductory
`tutorial <http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html>`__
of SSD, you may have better idea how it works. You can also refer to
original paper and entry level tutorials for idea that support SSD.

GluonVision has a model zoo which has a lot of built-in SSD networks.
Therefore you can simply load them from model\_zoo module like this:

.. code:: python

    from gluonvision import model_zoo
    net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)
    print(net)


.. parsed-literal::

    SSD(
      (class_predictors): HybridSequential(
        (0): ConvPredictor(
          (predictor): Conv2D(None -> 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): ConvPredictor(
          (predictor): Conv2D(None -> 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (2): ConvPredictor(
          (predictor): Conv2D(None -> 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (3): ConvPredictor(
          (predictor): Conv2D(None -> 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (4): ConvPredictor(
          (predictor): Conv2D(None -> 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (5): ConvPredictor(
          (predictor): Conv2D(None -> 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (box_predictors): HybridSequential(
        (0): ConvPredictor(
          (predictor): Conv2D(None -> 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): ConvPredictor(
          (predictor): Conv2D(None -> 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (2): ConvPredictor(
          (predictor): Conv2D(None -> 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (3): ConvPredictor(
          (predictor): Conv2D(None -> 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (4): ConvPredictor(
          (predictor): Conv2D(None -> 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (5): ConvPredictor(
          (predictor): Conv2D(None -> 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (cls_decoder): MultiPerClassDecoder(

      )
      (features): VGGAtrousExtractor(
        (norm4): Normalize(

        )
        (extras): HybridSequential(
          (0): HybridSequential(
            (0): Conv2D(None -> 256, kernel_size=(1, 1), stride=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (3): Activation(relu)
          )
          (1): HybridSequential(
            (0): Conv2D(None -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (3): Activation(relu)
          )
          (2): HybridSequential(
            (0): Conv2D(None -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 256, kernel_size=(3, 3), stride=(1, 1))
            (3): Activation(relu)
          )
          (3): HybridSequential(
            (0): Conv2D(None -> 128, kernel_size=(1, 1), stride=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 256, kernel_size=(3, 3), stride=(1, 1))
            (3): Activation(relu)
          )
        )
        (stages): HybridSequential(
          (0): HybridSequential(
            (0): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Activation(relu)
          )
          (1): HybridSequential(
            (0): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Activation(relu)
          )
          (2): HybridSequential(
            (0): Conv2D(None -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Activation(relu)
            (4): Conv2D(None -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (5): Activation(relu)
          )
          (3): HybridSequential(
            (0): Conv2D(None -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Activation(relu)
            (4): Conv2D(None -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (5): Activation(relu)
          )
          (4): HybridSequential(
            (0): Conv2D(None -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Activation(relu)
            (2): Conv2D(None -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Activation(relu)
            (4): Conv2D(None -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (5): Activation(relu)
          )
          (5): HybridSequential(
            (0): Conv2D(None -> 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
            (1): Activation(relu)
            (2): Conv2D(None -> 1024, kernel_size=(1, 1), stride=(1, 1))
            (3): Activation(relu)
          )
        )
      )
      (anchor_generators): HybridSequential(
        (0): SSDAnchorGenerator(

        )
        (1): SSDAnchorGenerator(

        )
        (2): SSDAnchorGenerator(

        )
        (3): SSDAnchorGenerator(

        )
        (4): SSDAnchorGenerator(

        )
        (5): SSDAnchorGenerator(

        )
      )
      (bbox_decoder): NormalizedBoxCenterDecoder(

      )
    )


SSD network is a HybridBlock as mentioned before. So you can call it
with an input as simple as:

.. code:: python

    import mxnet as mx
    x = mx.nd.zeros(shape=(1, 3, 300, 300))
    net.initialize()
    cids, scores, bboxes = net(x)

where ``cids`` is the class labels, ``scores`` are confidences of each
predictions, ``bboxes`` are corresponding bounding boxes' absolute
coordinates.

Training targets
----------------

Unlike a single ``SoftmaxCrossEntropyLoss`` used in image
classification, the losses used in SSD is more complicated. Don't worry
though, because we have these modules available out of box.

Checkout the ``target_generator`` in SSD networks.

.. code:: python

    print(net.target_generator)


.. parsed-literal::

    SSDTargetGenerator(
      (_sampler): OHEMSampler(

      )
      (_center_to_corner): BBoxCenterToCorner(

      )
      (_box_encoder): NormalizedBoxCenterEncoder(
        (corner_to_center): BBoxCornerToCenter(

        )
      )
      (_cls_encoder): MultiClassEncoder(

      )
      (_matcher): CompositeMatcher(

      )
    )


You can see there are bounding boxes encoder which transfers raw
coordinates to bbox prediction targets, a class encoder which generates
class labels for each anchor box. Matcher and samplers included are used
to apply various advanced strategies described in paper.

References
----------

[1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott
Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox
Detector. ECCV 2016.

Dive deep into training script
------------------------------

We include an training script for reproducing SOTA models.

Example:

::

    python train_ssd.py --network vgg16_atrous --data-shape 300 --batch-size 32 --gpus 0,1,2,3 -j 32 --log-interval 20

"""

import argparse
import os
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from gluonvision import data as gdata
from gluonvision import utils as gutils
from gluonvision.model_zoo import get_model
from gluonvision.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluonvision.data.transforms.presets.ssd import SSDDefaultValTransform
from gluonvision.utils.metrics.voc_detection import VOC07MApMetric
from gluonvision.utils.metrics.accuracy import Accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epoches at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    args = parser.parse_args()
    return args

def get_dataset(dataset):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset

def get_dataloader(train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    train_loader = gdata.DetectionDataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height)),
        batch_size, True, last_batch='rollover', num_workers=num_workers)
    val_loader = gdata.DetectionDataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, classes):
    """Test on validation dataset."""
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            bboxes = bboxes.clip(0, batch[0].shape[2])
            gt_ids = y.slice_axis(axis=-1, begin=4, end=5)
            gt_bboxes = y.slice_axis(axis=-1, begin=0, end=4)
            gt_difficults = y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None
            metric.update(bboxes, ids, scores, gt_bboxes, gt_ids, gt_difficults)

    return metric.get()

def train(net, train_data, val_data, classes, args):
    """Training pipeline"""
    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    # lr decay policy
    lr_decay = args.lr_decay
    lr_steps = [float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()]
    if not isinstance(lr_decay, list):
        lr_decay = [lr_decay]
    if len(lr_decay) == 1 and len(lr_steps) > 1:
        lr_decay *= len(lr_steps)

    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    box_loss = gluon.loss.HuberLoss()
    acc_metric = Accuracy(axis=-1, ignore_labels=[-1])
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    logging.info('Start training from [Epoch %d]' % args.start_epoch)
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in lr_steps:
            new_lr = trainer.learning_rate * np.prod(lr_decay[:lr_steps.index(epoch)])
            trainer.set_learning_rate(new_lr)
            logging.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        acc_metric.reset()
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            labels = []
            losses1 = []
            losses2 = []
            losses3 = []
            losses4 = []
            Ls = []
            num_positive = []
            with autograd.record():
                for x, y in zip(data, label):
                    cls_preds, box_preds, anchors = net(x)
                    with autograd.pause():
                        gt_boxes = nd.slice_axis(y, axis=-1, begin=0, end=4)
                        gt_ids = nd.slice_axis(y, axis=-1, begin=4, end=5)
                        cls_targets, box_targets, box_masks = net.target_generator(
                            anchors, cls_preds, gt_boxes, gt_ids)
                        num_positive.append(nd.sum(cls_targets > 0).asscalar())

                    l1 = cls_loss(cls_preds, cls_targets, (cls_targets >= 0).expand_dims(axis=-1))
                    losses3.append(l1 * cls_targets.size / cls_targets.shape[0])
                    l2 = box_loss(box_preds * box_masks, box_targets)
                    losses4.append(l2 * box_targets.size / box_targets.shape[0])
                    outputs.append(cls_preds)
                    labels.append(cls_targets)
                n_pos = max(1, sum(num_positive))
                for l3, l4 in zip(losses3, losses4):
                    L = l3 / n_pos + l4 / n_pos
                    Ls.append(L)
                    losses1.append(l3 / n_pos * batch_size)  # rescale for batch
                    losses2.append(l4 / n_pos * batch_size)  # rescale for batch
                autograd.backward(Ls)
            trainer.step(1)
            ce_metric.update(0, losses1)
            smoothl1_metric.update(0, losses2)
            acc_metric.update(labels, outputs)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                name3, loss3 = acc_metric.get()
                logging.info('[Epoch %d][Batch %d], Speed: %f samples/sec, %s=%f, %s=%f, %s=%f'%(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        name3, loss3 = acc_metric.get()
        logging.info('[Epoch %d] Training cost: %f, %s=%f, %s=%f, %s=%f'%(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3))
        map_name, mean_ap = validate(net, val_data, ctx, classes)
        val_msg = '\n'.join(['%s=%f'%(k, v) for k, v in zip(map_name, mean_ap)])
        logging.info('[Epoch %d] Validation: \n%s'%(epoch, val_msg))
        save_params(net, best_map, mean_ap[-1], epoch, args.save_interval, args.save_prefix)

if __name__ == '__main__':
    args = parse_args()
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # training data
    train_dataset, val_dataset = get_dataset(args.dataset)
    train_data, val_data = get_dataloader(
        train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = train_dataset.classes  # class names

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    net = get_model(net_name, pretrained_base=True)
    if args.resume.strip():
        net.load_params(args.resume.strip())

    # training
    args.save_prefix += net_name
    train(net, train_data, val_data, classes, args)
