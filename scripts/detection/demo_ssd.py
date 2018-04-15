"""
Getting started with SSD pre-trained models
===========================================

This article is an introductory tutorial to play with pre-trained models
with several lines of code.

For us to begin with, mxnet and gluonvvision modules are required to be
installed.

A quick solution is

::

    pip install --user --pre mxnet gluonvision

or please refer to offical `installation
guide <http://gluon-vision.mxnet.io.s3-website-us-west-2.amazonaws.com/index.html#installation>`__.

.. code:: ipython3

    import mxnet as mx
    import gluonvision as gv
    from matplotlib import pyplot as plt
    %matplotlib inline

Obtain a with pretrained model
------------------------------

In this section, we grab a pretrained model and test image to play with

.. code:: ipython3

    # try grab a 300x300 model trained on Pascal voc dataset.
    # it will automatically download from s3 servers if not exists
    net = gv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

Pre-process image
-----------------

A raw image must be converted to tensor before inference.

.. code:: ipython3

    from gluonvision.data.transforms import presets

    # a demo image, feel free to use your own image
    image_name = 'street.jpg'

    # gluonvision support SSD models to accept arbitrary data shape
    # so this size is not that restrictive
    x, img = presets.ssd.load_test(image_name, short=512)
    print('shape of x:', x.shape)


.. parsed-literal::

    shape of x: (1, 3, 512, 512)


Inference and display
---------------------

.. code:: ipython3

    ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
    from gluonvision.utils import viz
    ax = viz.plot_bbox(img, bboxes, scores, ids, class_names=net.classes, ax=None)
    plt.show()



.. image:: https://github.com/zhreshold/gluonvision-tutorials/raw/master/detection/ssd/output_8_0.png

Dive Deep into Source Codes
---------------------------

This script allows you to download or use local pretrained model to test
and display one/multiple image(s).

Example usages:

::

    python ssd_demo.py --network ssd_512_resnet50_v1_voc --images street.jpg --gpus 0

"""
import os
import argparse
import mxnet as mx
import gluonvision as gv
from gluonvision.data.transforms import presets
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test with SSD networks.')
    parser.add_argument('--network', type=str, default='ssd_300_vgg16_atrous_voc',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    if not args.images.strip():
        gv.utils.download("https://cloud.githubusercontent.com/assets/3307514/" +
            "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg", 'street.jpg')
        image_list = ['street.jpg']
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gv.model_zoo.get_model(args.network, pretrained=False)
        net.load_params(args.pretrained)
    net.set_nms(0.45, 200)

    ax = None
    for image in image_list:
        x, img = presets.ssd.load_test(image, short=512)
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        ax = gv.utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                                    class_names=net.classes, ax=ax)
        plt.show()
