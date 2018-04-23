Vision Model Zoo
================

The GluonVision Model Zoo,
similar to its upstream `Gluon Model Zoo
<https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_,
provides pre-defined and pre-trained models to help bootstrap computer vision related applications.

Model Zoo API
-------------

.. code-block:: python

    from gluonvision import model_zoo
    cifar_resnet20 = model_zoo.get_model('cifar_resnet20_v1', pretrained=True)
    # a fully pretrained ssd model
    ssd0 = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=True)
    # load ssd model with pretrained feature extractors
    ssd1 = model_zoo.get_model('ssd_512_vgg16_atrous_voc', pretrained_base=True)
    # load ssd from scratch
    ssd2 = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained_base=False)

We recommend using :py:meth:`gluonvision.model_zoo.get_model` for loading pre-defined models, because it provides
name check and suggest you what models are available now.

However, you can still load models by directly instantiate it like

.. code-block:: python

    from gluonvision import model_zoo
    cifar_resnet20 = model_zoo.cifar_resnet20_v1(pretrained=True)

.. hint::

  Detailed ``model_zoo`` APIs are available in our API reference: :py:meth:`gluonvision.model_zoo`.

Summary of Available Models
---------------------------

We are still in early development stage, more models will be made available for download.

Image Classification
~~~~~~~~~~~~~~~~~~~~

The following table summarizes the available models and their performances in additional to
`Gluon Model Zoo
<https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_.

+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Model                     | Accuracy (Vanilla) | Accuracy (Mix-Up) [4]_ | Training Command  | Training Log     |
+===========================+====================+===================+===================+==================+
| CIFAR_ResNet20_v1    [1]_ | 90.8               | 91.6              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v1_mixup.sh>`_         | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v1_mixup.log>`_     |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet56_v1    [1]_ | 92.8               | 93.8              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet56_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet56_v1_mixup.sh>`_         | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet56_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v1_mixup.log>`_    |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet110_v1   [1]_ | 93.4               | 94.7              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v1_mixup.sh>`_       | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v1_mixup.log>`_    |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet20_v2    [2]_ | 90.8               | 91.3              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v2_mixup.sh>`_         | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet20_v2_mixup.log>`_     |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet56_v2    [2]_ | 93.1               | 94.1              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet56_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet56_v2_mixup.sh>`_         | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet56_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet56_v2_mixup.log>`_     |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet110_v2   [2]_ | 93.7               | 94.6              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v2_mixup.sh>`_       | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_resnet110_v2_mixup.log>`_    |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_WideResNet16_10 [3]_ | 95.1               | 96.1              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet16_10.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet16_10_mixup.sh>`_ | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet16_10.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet16_10_mixup.log>`_ |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_WideResNet28_10 [3]_ | 95.6               | 96.6              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet28_10.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet28_10_mixup.sh>`_ | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet28_10.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet28_10_mixup.log>`_ |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_WideResNet40_8  [3]_ | 95.9               | 96.7              | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet40_8.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet40_8_mixup.sh>`_   | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet40_8.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/classification/cifar/cifar_wideresnet40_8_mixup.log>`_  |
+---------------------------+--------------------+-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Object Detection
~~~~~~~~~~~~~~~~

The following table summarizes the available models and their performances for object detection.

.. https://bit.ly/2qQHLl4

.. hint::

  The detection model names contain information for you to recognize them.

  For instance, ``ssd_300_vgg16_atrous_voc`` consists of four parts,

  - ``ssd`` indicate the algorithm is "Single Shot Multibox Object Detection" [5]_.

  - ``300`` is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this spatial shape.

  - ``vgg16_atrous`` is the name of base feature extractor network.

  - ``voc`` is the training dataset.

+------------------------------------+------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| Model                              | mAP  | Training Command                                                                                                                     | Training log                                                                                                                        |
+====================================+======+======================================================================================================================================+=====================================================================================================================================+
| ssd_300_vgg16_atrous_voc [5]_      | 77.6 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/detection/ssd_300_vgg16_atrous_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/detection/ssd_300_vgg16_atrous_voc_train.log>`_       |
+------------------------------------+------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_vgg16_atrous_voc [5]_      | 79.2 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/detection/ssd_512_vgg16_atrous_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/detection/ssd_512_vgg16_atrous_voc_train.log>`_       |
+------------------------------------+------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_resnet50_v1_voc [5]_       | 80.1 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/logs/detection/ssd_512_resnet50_v1_voc.sh>`_       |                                                                                                                                     |
+------------------------------------+------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+



Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~

Table of pre-trained models, performances and the training commands:

.. comment (models :math:`^\ast` denotes pre-trained on COCO):

.. role:: raw-html(raw)
   :format: html

+-------------------+--------------+------------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+
| Name              | Method       | Backbone   | Dataset   | Note      | mIoU      | Command                                                                                      |
+===================+==============+============+===========+===========+===========+==============================================================================================+
| fcn_resnet50_voc  | FCN [6]_     | ResNet50   | PASCAL12  | stride 8  | 69.4_     | :raw-html:`<a href="javascript:toggleblock('cmd_fcn_50')" class="toggleblock">cmd</a>`       |
+-------------------+--------------+------------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+
| fcn_resnet101_voc | FCN [6]_     | ResNet101  | PASCAL12  | stride 8  | 70.9_     | :raw-html:`<a href="javascript:toggleblock('cmd_fcn_101')" class="toggleblock">cmd</a>`      |
+-------------------+--------------+------------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+

.. _69.4:  http://host.robots.ox.ac.uk:8080/anonymous/TC12D2.html
.. _70.9:  http://host.robots.ox.ac.uk:8080/anonymous/FTIQXJ.html

.. raw:: html

    <code xml:space="preserve" id="cmd_fcn_50" style="display: none; text-align: left; white-space: pre-wrap">
    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_aug --model fcn --backbone resnet50 --lr 0.001 --syncbn --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --model fcn --backbone resnet50 --lr 0.0001 --syncbn --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params
    </code>

    <code xml:space="preserve" id="cmd_fcn_101" style="display: none; text-align: left; white-space: pre-wrap">
    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_aug --model fcn --backbone resnet101 --lr 0.001 --syncbn --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --model fcn --backbone resnet101 --lr 0.0001 --syncbn --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params
    </code>

.. [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Deep residual learning for image recognition." \
       In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
.. [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Identity mappings in deep residual networks." \
       In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.
.. [3] Zagoruyko, Sergey, and Nikos Komodakis. \
       "Wide residual networks." \
       arXiv preprint arXiv:1605.07146 (2016).
.. [4] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. \
       "mixup: Beyond empirical risk minimization." \
       arXiv preprint arXiv:1710.09412 (2017).
.. [5] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
.. [6] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. \
    "Fully convolutional networks for semantic segmentation." \
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
