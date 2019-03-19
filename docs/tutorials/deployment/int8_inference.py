"""3. Inference with Quantized Models
=====================================

This is a tutorial which illustrates how to use quantized GluonCV
models for inference on Intel Xeon Processors to gain higher performance.

The following example requires ``GluonCV>=0.4`` and ``MXNet-mkl>=1.5.0b20190314``. Please follow `our installation guide <../../index.html#installation>`__ to install or upgrade GluonCV and nightly build of MXNet if necessary.

Introduction
------------

GluonCV delivered some quantized models to improve the performance and reduce the deployment costs for the computer vision inference tasks. In real production, there are two main benefits of lower precision (INT8). First, the computation can be accelerated by the low precision instruction, like Intel Vector Neural Network Instruction (VNNI). Second, lower precision data type would save the memory bandwidth and allow for better cache locality and save the power. The new feature can get up to 2X performance speedup in the current AWS EC2 CPU instances and will reach 4X under the `Intel Deep Learning Boost (VNNI) <https://www.intel.ai/intel-deep-learning-boost/#gs.0ngn54>`_ enabled hardware with less than 0.5% accuracy drop.

Please checkout `verify_pretrained.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/classification/imagenet/verify_pretrained.py>`_ for imagenet inference
and `eval_ssd.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/detection/ssd/eval_ssd.py>`_ for SSD inference.

Performance
-----------

GluonCV supports some quantized classification models and detection models.
For the throughput, the target is to achieve the maximum machine efficiency to combine the inference requests together and get the results by one iteration. From the bar-chart, it is clearly that the quantization approach improved the throughput from 1.46X to 2.71X for selected models.
Below CPU performance is from AWS EC2 C5.18xlarge with 18 cores.

.. figure:: https://user-images.githubusercontent.com/17897736/54540947-dc08c480-49d3-11e9-9a0d-a97d44f9792c.png
   :alt: Gluon Quantization Performance

   Gluon Quantization Performance

+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
|  Model                | Dataset  | Batch Size | C5.18xlarge FP32 | C5.18xlarge INT8 | Speedup | FP32 Accuracy   | INT8 Accuracy   |
+=======================+==========+============+==================+==================+=========+=================+=================+
| ResNet50 V1           | ImageNet | 128        | 122.02           | 276.72           | 2.27    | 77.21%/93.55%   | 76.86%/93.46%   |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| MobileNet 1.0         | ImageNet | 128        | 375.33           | 1016.39          | 2.71    | 73.28%/91.22%   | 72.85%/90.99%   |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-VGG 300*          | VOC      | 224        | 21.55            | 31.47            | 1.46    | 77.4            | 77.46           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-VGG 512*          | VOC      | 224        | 7.63             | 11.69            | 1.53    | 78.41           | 78.39           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-resnet50_v1 512*  | VOC      | 224        | 17.81            | 34.55            | 1.94    | 80.21           | 80.16           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-mobilenet1.0 512* | VOC      | 224        | 31.13            | 48.72            | 1.57    | 75.42           | 75.04           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+

Quantized SSD models are evaluated with ``nms_thresh=0.45``, ``nms_topk=200``.

Demo usage for SSD
------------------

.. code:: bash

   # with Pascal VOC validation dataset saved on disk
   python eval_ssd.py --network=vgg16_atrous --quantized --data-shape=300 --batch-size=224 --dataset=voc

Usage:

::

   SYNOPSIS
            python eval_ssd.py [-h] [--network NETWORK] [--quantized]
                               [--data-shape DATA_SHAPE] [--batch-size BATCH_SIZE]
                               [--dataset DATASET] [--num-workers NUM_WORKERS]
                               [--num-gpus NUM_GPUS] [--pretrained PRETRAINED]
                               [--save-prefix SAVE_PREFIX]

   OPTIONS
            -h, --help            show this help message and exit
            --network NETWORK     Base network name
            --quantized           use int8 pretrained model
            --data-shape DATA_SHAPE
                                    Input data shape
            --batch-size BATCH_SIZE
                                    eval mini-batch size
            --dataset DATASET     eval dataset.
            --num-workers NUM_WORKERS, -j NUM_WORKERS
                                    Number of data workers
            --num-gpus NUM_GPUS   number of gpus to use.
            --pretrained PRETRAINED
                                    Load weights from previously saved parameters.
            --save-prefix SAVE_PREFIX
                                    Saving parameter prefix
"""
