.. _gluoncv-model-zoo-classification:

Classification
====================



Visualization of Inference Throughputs vs. Validation Accuracy of ImageNet pre-trained models is illustrated in the following graph. Throughputs are measured with single V100 GPU and batch size 64.

.. image:: /_static/plot_help.png
  :width: 100%

.. include:: /_static/classification_throughputs.html

How To Use Pretrained Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The following example requires ``GluonCV>=0.4`` and ``MXNet>=1.4.0``. Please follow `our installation guide <../index.html#installation>`__ to install or upgrade GluonCV and MXNet if necessary.

- Prepare an image by yourself or use `our sample image <../_static/classification-demo.png>`__. You can save the image into filename ``classification-demo.png`` in your working directory or change the filename in the source codes if you use an another name.

- Use a pre-trained model. A model is specified by its name.

Let's try it out!

.. code-block:: python

    import mxnet as mx
    import gluoncv

    # you can change it to your image filename
    filename = 'classification-demo.png'
    # you may modify it to switch to another model. The name is case-insensitive
    model_name = 'ResNet50_v1d'
    # download and load the pre-trained model
    net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    # load image
    img = mx.image.imread(filename)
    # apply default data preprocessing
    transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
    # run forward pass to obtain the predicted score for each class
    pred = net(transformed_img)
    # map predicted values to probability by softmax
    prob = mx.nd.softmax(pred)[0].asnumpy()
    # find the 5 class indices with the highest score
    ind = mx.nd.topk(pred, k=5)[0].astype('int').asnumpy().tolist()
    # print the class name and predicted probability
    print('The input picture is classified to be')
    for i in range(5):
        print('- [%s], with probability %.3f.'%(net.classes[ind[i]], prob[ind[i]]))

The output from `our sample image <../_static/classification-demo.png>`__ is expected to be

.. code-block:: txt

    The input picture is classified to be
    - [Welsh springer spaniel], with probability 0.899.
    - [Irish setter], with probability 0.005.
    - [Brittany spaniel], with probability 0.003.
    - [cocker spaniel], with probability 0.002.
    - [Blenheim spaniel], with probability 0.002.


Remember, you can try different models by replacing the value of ``model_name``.
Read further for model names and their performances in the tables.

.. role:: greytag

ImageNet
~~~~~~~~

.. hint::

    Training commands work with this script:

    :download:`Download train_imagenet.py<../../scripts/classification/imagenet/train_imagenet.py>`

    A model can have differently trained parameters with different hashtags.
    Parameters with :greytag:`a grey name` can be downloaded by passing the corresponding hashtag.

    - Download default pretrained weights: ``net = get_model('ResNet50_v1d', pretrained=True)``

    - Download weights given a hashtag: ``net = get_model('ResNet50_v1d', pretrained='117a384e')``

    ``ResNet50_v1_int8`` and ``MobileNet1.0_int8`` are quantized model calibrated on ImageNet dataset.

.. role:: tag

ResNet
------

.. hint::

    - ``ResNet50_v1_int8`` is a quantized model for ``ResNet50_v1``.

    - ``ResNet_v1b`` modifies ``ResNet_v1`` by setting stride at the 3x3 layer for a bottleneck block.

    - ``ResNet_v1c`` modifies ``ResNet_v1b`` by replacing the 7x7 conv layer with three 3x3 conv layers.

    - ``ResNet_v1d`` modifies ``ResNet_v1c`` by adding an avgpool layer 2x2 with stride 2 downsample feature map on the residual path to preserve more information.

.. table::
   :widths: 45 5 5 10 20 15

   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                     | Top-1  | Top-5  | Hashtag  | Training Command                                                                                                                     | Training Log                                                                                                                  |
   +===========================+========+========+==========+======================================================================================================================================+===============================================================================================================================+
   | ResNet18_v1 [1]_          | 70.93  | 89.92  | a0666292 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1.log>`_          |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet34_v1 [1]_          | 74.37  | 91.87  | 48216ba9 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1.log>`_          |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet50_v1 [1]_          | 77.36  | 93.57  | cc729d95 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1.log>`_          |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet50_v1_int8 [1]_     | 76.86  | 93.46  | cc729d95 |                                                                                                                                      |                                                                                                                               |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet101_v1 [1]_         | 78.34  | 94.01  | d988c13d | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet152_v1 [1]_         | 79.22  | 94.64  | acfd0970 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet18_v1b [1]_         | 70.94  | 89.83  | 2d9d980c | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1b.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1b.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet34_v1b [1]_         | 74.65  | 92.08  | 8e16b848 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1b.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1b.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet50_v1b [1]_         | 77.67  | 93.82  | 0ecdba34 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1b.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1b.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet50_v1b_gn [1]_      | 77.36  | 93.59  | 0ecdba34 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1b_gn.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1b_gn.log>`_      |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet101_v1b [1]_        | 79.20  | 94.61  | a455932a | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1b.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1b.log>`_        |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet152_v1b [1]_        | 79.69  | 94.74  | a5a61ee1 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1b.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1b.log>`_        |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet50_v1c [1]_         | 78.03  | 94.09  | 2a4e0708 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1c.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1c.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet101_v1c [1]_        | 79.60  | 94.75  | 064858f2 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1c.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1c.log>`_        |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet152_v1c [1]_        | 80.01  | 94.96  | 75babab6 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1b.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1b.log>`_        |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet50_v1d [1]_         | 79.15  | 94.58  | 117a384e | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1d-mixup.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1d-mixup.log>`_   |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | :tag:`ResNet50_v1d` [1]_  | 78.48  | 94.20  | 00319ddc | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1d.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1d.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet101_v1d [1]_        | 80.51  | 95.12  | 1b2b825f | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1d-mixup.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1d-mixup.log>`_  |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | :tag:`ResNet101_v1d` [1]_ | 79.78  | 94.80  | 8659a9d6 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1d.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1d.log>`_        |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet152_v1d [1]_        | 80.61  | 95.34  | cddbc86f | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1d-mixup.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1d-mixup.log>`_  |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | :tag:`ResNet152_v1d` [1]_ | 80.26  | 95.00  | cfe0220d | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1d.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1d.log>`_        |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet18_v2 [2]_          | 71.00  | 89.92  | a81db45f | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v2.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v2.log>`_          |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet34_v2 [2]_          | 74.40  | 92.08  | 9d6b80bb | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v2.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v2.log>`_          |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet50_v2 [2]_          | 77.11  | 93.43  | ecdde353 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v2.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v2.log>`_          |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet101_v2 [2]_         | 78.53  | 94.17  | 18e93e4f | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v2.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v2.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNet152_v2 [2]_         | 79.21  | 94.31  | f2695542 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v2.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v2.log>`_         |
   +---------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

ResNext
-------

.. table::
   :widths: 45 5 5 10 20 15

   +---------------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                           | Top-1  | Top-5  | Hashtag  | Training Command                                                                                                                     | Training Log                                                                                                                  |
   +=================================+========+========+==========+======================================================================================================================================+===============================================================================================================================+
   | ResNext50_32x4d [12]_           | 79.32  | 94.53  | 4ecf62e2 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnext50_32x4d.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnext50_32x4d.log>`_      |
   +---------------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNext101_32x4d [12]_          | 80.37  | 95.06  | 8654ca5d | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnext101_32x4d.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnext101_32x4d.log>`_     |
   +---------------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | ResNext101_64x4d_v1 [12]_       | 80.69  | 95.17  | 2f0d1c9d | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnext101_64x4d.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnext101_64x4d.log>`_     |
   +---------------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | SE_ResNext50_32x4d [12]_ [14]_  | 79.95  | 94.93  | 7906e0e1 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/se_resnext50_32x4d.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/se_resnext50_32x4d.log>`_   |
   +---------------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | SE_ResNext101_32x4d [12]_ [14]_ | 80.91  | 95.39  | 688e2389 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/se_resnext101_32x4d.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/se_resnext101_32x4d.log>`_  |
   +---------------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | SE_ResNext101_64x4d [12]_ [14]_ | 81.01  | 95.32  | 11c50114 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/se_resnext101_64x4d.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/se_resnext101_64x4d.log>`_  |
   +---------------------------------+--------+--------+----------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

MobileNet
---------

.. hint::

    - ``MobileNet1.0_int8`` is a quantized model for ``MobileNet1.0``.

.. table::
   :widths: 45 5 5 10 20 15

   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                    | Top-1  | Top-5  | Hashtag  | Training Command                                                                                                                        | Training Log                                                                                                                  |
   +==========================+========+========+==========+=========================================================================================================================================+===============================================================================================================================+
   | MobileNet1.0 [4]_        | 73.28  | 91.30  | efbb2ca3 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet1.0-mixup.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet1.0-mixup.log>`_   |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNet1.0_int8 [4]_   | 72.85  | 90.99  | efbb2ca3 |                                                                                                                                         |                                                                                                                               |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | :tag:`MobileNet1.0` [4]_ | 72.93  | 91.14  | cce75496 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet1.0.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet1.0.log>`_         |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNet0.75 [4]_       | 70.25  | 89.49  | 84c801e2 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet0.75.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet0.75.log>`_        |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNet0.5 [4]_        | 65.20  | 86.34  | 0130d2aa | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet0.5.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet0.5.log>`_         |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNet0.25 [4]_       | 52.91  | 76.94  | f0046a3d | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet0.25.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenet0.25.log>`_        |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNetV2_1.0 [5]_     | 72.04  | 90.57  | f9952bcd | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_1.0.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_1.0.log>`_      |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNetV2_0.75 [5]_    | 69.36  | 88.50  | b56e3d1c | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.75.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.75.log>`_     |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNetV2_0.5 [5]_     | 64.43  | 85.31  | 08038185 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.5.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.5.log>`_      |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNetV2_0.25 [5]_    | 51.76  | 74.89  | 9b1d2cc3 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.25.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.25.log>`_     |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNetV3_Large [15]_  | 75.32  | 92.30  | eaa44578 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv3_large.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv3_large.log>`_    |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | MobileNetV3_Small [15]_  | 67.72  | 87.51  | 33c100a7 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv3_small.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv3_small.log>`_    |
   +--------------------------+--------+--------+----------+-----------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

VGG
---

.. table::
   :widths: 45 5 5 10 20 15

   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                 | Top-1  | Top-5  | Hashtag  | Training Command                                                                                                                   | Training Log                                                                                                                  |
   +=======================+========+========+==========+====================================================================================================================================+===============================================================================================================================+
   | VGG11 [9]_            | 66.62  | 87.34  | dd221b16 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | VGG13 [9]_            | 67.74  | 88.11  | 6bc5de58 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | VGG16 [9]_            | 73.23  | 91.31  | e660d456 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg16.sh>`_             | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg16.log>`_                |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | VGG19 [9]_            | 74.11  | 91.35  | ad2f660d | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg19.sh>`_             | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg19.log>`_                |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | VGG11_bn [9]_         | 68.59  | 88.72  | ee79a809 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | VGG13_bn [9]_         | 68.84  | 88.82  | 7d97a06c |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | VGG16_bn [9]_         | 73.10  | 91.76  | 7f01cf05 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg16_bn.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg16_bn.log>`_             |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | VGG19_bn [9]_         | 74.33  | 91.85  | f360b758 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg19_bn.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/vgg19_bn.log>`_             |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

SqueezeNet
----------

.. table::
   :widths: 45 5 5 10 20 15

   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                 | Top-1  | Top-5  | Hashtag  | Training Command                                                                                                                   | Training Log                                                                                                                  |
   +=======================+========+========+==========+====================================================================================================================================+===============================================================================================================================+
   | SqueezeNet1.0 [10]_   | 56.11  | 79.09  | 264ba497 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | SqueezeNet1.1 [10]_   | 54.96  | 78.17  | 33ba0f93 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

DenseNet
--------

.. table::
   :widths: 45 5 5 10 20 15

   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                 | Top-1  | Top-5  | Hashtag  | Training Command                                                                                                                   | Training Log                                                                                                                  |
   +=======================+========+========+==========+====================================================================================================================================+===============================================================================================================================+
   | DenseNet121 [7]_      | 74.97  | 92.25  | f27dbf2d |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | DenseNet161 [7]_      | 77.70  | 93.80  | b6c8a957 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | DenseNet169 [7]_      | 76.17  | 93.17  | 2603f878 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | DenseNet201 [7]_      | 77.32  | 93.62  | 1cdbc116 |                                                                                                                                    |                                                                                                                               |
   +-----------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

Pruned ResNet
-------------

.. table::
   :widths: 45 5 5 10 35

   +----------------------+-------+-------+----------+------------------------------+
   | Model                | Top-1 | Top-5 | Hashtag  | Speedup (to original ResNet) |
   +======================+=======+=======+==========+==============================+
   | resnet18_v1b_0.89    | 67.2  | 87.45 | 54f7742b | 2x                           |
   +----------------------+-------+-------+----------+------------------------------+
   | resnet50_v1d_0.86    | 78.02 | 93.82 | a230c33f | 1.68x                        |
   +----------------------+-------+-------+----------+------------------------------+
   | resnet50_v1d_0.48    | 74.66 | 92.34 | 0d3e69bb | 3.3x                         |
   +----------------------+-------+-------+----------+------------------------------+
   | resnet50_v1d_0.37    | 70.71 | 89.74 | 9982ae49 | 5.01x                        |
   +----------------------+-------+-------+----------+------------------------------+
   | resnet50_v1d_0.11    | 63.22 | 84.79 | 6a25eece | 8.78x                        |
   +----------------------+-------+-------+----------+------------------------------+
   | resnet101_v1d_0.76   | 79.46 | 94.69 | a872796b | 1.8x                         |
   +----------------------+-------+-------+----------+------------------------------+
   | resnet101_v1d_0.73   | 78.89 | 94.48 | 712fccb1 | 2.02x                        |
   +----------------------+-------+-------+----------+------------------------------+


Others
------

.. hint::

    ``InceptionV3`` is evaluated with input size of 299x299.

.. table::
   :widths: 45 5 5 10 20 15

   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                   | Top-1  | Top-5  | Hashtag  | Training Command                                                                                                                   | Training Log                                                                                                                  |
   +=========================+========+========+==========+====================================================================================================================================+===============================================================================================================================+
   | AlexNet [6]_            | 54.92  | 78.03  | 44335d1f |                                                                                                                                    |                                                                                                                               |
   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | darknet53   [3]_        | 78.56  | 94.43  | 2189ea49 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/darknet53-mixup.sh>`_   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/darknet53-mixup.log>`_      |
   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | :tag:`darknet53` [3]_   | 78.13  | 93.86  | 95975047 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/darknet53.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/darknet53.log>`_            |
   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | InceptionV3 [8]_        | 78.77  | 94.39  | a5050dbc | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/inceptionv3-mixup.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/inceptionv3-mixup.log>`_    |
   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Xception [8]_           | 79.56  | 94.77  | 37c1c90b | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/train_xception.sh>`_    |  `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/xception.log>`_            |
   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | :tag:`InceptionV3` [8]_ | 78.41  | 94.13  | e132adf2 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/inceptionv3.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/inceptionv3.log>`_          |
   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | SENet_154 [14]_         | 81.26  | 95.51  | b5538ef1 |                                                                                                                                    |                                                                                                                               |
   +-------------------------+--------+--------+----------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

CIFAR10
~~~~~~~

The following table lists pre-trained models trained on CIFAR10.

.. hint::

    Our pre-trained models reproduce results from "Mix-Up" [13]_ .
    Please check the reference paper for further information.

    Training commands in the table work with the following scripts:

    - For vanilla training: :download:`Download train_cifar10.py<../../scripts/classification/cifar/train_cifar10.py>`
    - For mix-up training: :download:`Download train_mixup_cifar10.py<../../scripts/classification/cifar/train_mixup_cifar10.py>`

.. table::
   :widths: 40 15 25 30

   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Model                        | Acc (Vanilla/Mix-Up [13]_ )| Training Command                                                                                                                                                                                                                                                         | Training Log                                                                                                                                                                                                                                                              |
   +==============================+============================+==========================================================================================================================================================================================================================================================================+===========================================================================================================================================================================================================================================================================+
   | CIFAR_ResNet20_v1 [1]_       | 92.1 / 92.9                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/ dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1_mixup.log>`_           |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_ResNet56_v1 [1]_       | 93.6 / 94.2                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/ dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1_mixup.log>`_           |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_ResNet110_v1 [1]_      | 93.0 / 95.2                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1_mixup.sh>`_           | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1_mixup.log>`_          |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_ResNet20_v2 [2]_       | 92.1 / 92.7                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/ dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2_mixup.log>`_           |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_ResNet56_v2 [2]_       | 93.7 / 94.6                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/ dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2_mixup.log>`_           |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_ResNet110_v2 [2]_      | 94.3 / 95.5                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2_mixup.sh>`_           | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2_mixup.log>`_          |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_WideResNet16_10 [11]_  | 95.1 / 96.7                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10_mixup.sh>`_     | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10_mixup.log>`_    |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_WideResNet28_10 [11]_  | 95.6 / 97.2                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10_mixup.sh>`_     | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10.log>`_ / `Mix-Up <https:// raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10_mixup.log>`_   |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_WideResNet40_8 [11]_   | 95.9 / 97.3                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8_mixup.sh>`_       | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8.log>`_ / `Mix-Up <https:// raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8_mixup.log>`_     |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | CIFAR_ResNeXt29_16x64d [12]_ | 96.3 / 97.3                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d_mixup.sh>`_   | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d.log>`_ / `Mix-Up <https:// raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d_mixup.log>`_ |
   +------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Deep residual learning for image recognition." \
       In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
.. [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Identity mappings in deep residual networks." \
       In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.
.. [3] Redmon, Joseph, and Ali Farhadi. \
       "Yolov3: An incremental improvement." \
       arXiv preprint arXiv:1804.02767 (2018).
.. [4] Howard, Andrew G., Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, \
       Tobias Weyand, Marco Andreetto, and Hartwig Adam. \
       "Mobilenets: Efficient convolutional neural networks for mobile vision applications." \
       arXiv preprint arXiv:1704.04861 (2017).
.. [5] Sandler, Mark, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. \
       "Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation." \
       arXiv preprint arXiv:1801.04381 (2018).
.. [6] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. \
       "Imagenet classification with deep convolutional neural networks." \
       In Advances in neural information processing systems, pp. 1097-1105. 2012.
.. [7] Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. \
       "Densely Connected Convolutional Networks." In CVPR, vol. 1, no. 2, p. 3. 2017.
.. [8] Szegedy, Christian, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. \
       "Rethinking the inception architecture for computer vision." \
       In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826. 2016.
.. [9] Karen Simonyan, Andrew Zisserman. \
       "Very Deep Convolutional Networks for Large-Scale Image Recognition." \
       arXiv technical report arXiv:1409.1556 (2014).
.. [10] Iandola, Forrest N., Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. \
       "Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size." \
       arXiv preprint arXiv:1602.07360 (2016).
.. [11] Zagoruyko, Sergey, and Nikos Komodakis. \
       "Wide residual networks." \
       arXiv preprint arXiv:1605.07146 (2016).
.. [12] Xie, Saining, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. \
       "Aggregated residual transformations for deep neural networks." \
       In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on, pp. 5987-5995. IEEE, 2017.
.. [13] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. \
       "mixup: Beyond empirical risk minimization." \
       arXiv preprint arXiv:1710.09412 (2017).
.. [14] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." arXiv preprint arXiv:1709.01507 7 (2017).
.. [15] Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. \
       "Searching for mobilenetv3." \
       arXiv preprint arXiv:1905.02244 (2019).
