"""3. Inference with Quantized Models
=====================================

This is a tutorial which illustrates how to use quantized GluonCV
models for inference on Intel Xeon Processors to gain higher performance.

The following example requires ``GluonCV>=0.4`` and ``MXNet-mkl>=1.5.0b20190623``. Please follow `our installation guide <../../index.html#installation>`__ to install or upgrade GluonCV and nightly build of MXNet if necessary.

Introduction
------------

GluonCV delivered some quantized models to improve the performance and reduce the deployment costs for the computer vision inference tasks. In real production, there are two main benefits of lower precision (INT8). First, the computation can be accelerated by the low precision instruction, like Intel Vector Neural Network Instruction (VNNI). Second, lower precision data type would save the memory bandwidth and allow for better cache locality and save the power. The new feature can get up to 4X performance speedup in the latest `AWS EC2 C5 instances <https://aws.amazon.com/blogs/aws/now-available-new-c5-instance-sizes-and-bare-metal-instances/>`_ under the `Intel Deep Learning Boost (VNNI) <https://www.intel.ai/intel-deep-learning-boost/>`_ enabled hardware with less than 0.5% accuracy drop.

Please checkout `verify_pretrained.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/classification/imagenet/verify_pretrained.py>`_ for imagenet inference
and `eval_ssd.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/detection/ssd/eval_ssd.py>`_ for SSD inference.

Performance
-----------

GluonCV supports some quantized classification models and detection models.
For the throughput, the target is to achieve the maximum machine efficiency to combine the inference requests together and get the results by one iteration. From the bar-chart, it is clearly that the fusion and quantization approach improved the throughput from 3.22X to 7.24X for selected models.
Below CPU performance is collected with dummy input from AWS EC2 C5.24xlarge instance with 24 physical cores.

.. figure:: https://user-images.githubusercontent.com/17897736/60255306-62129200-9884-11e9-96de-3f145be70431.png
   :alt: Gluon Quantization Performance

   Gluon Quantization Performance

+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
|  Model                | Dataset  | Batch Size | C5.24xlarge FP32 | C5.24xlarge INT8 | Speedup | FP32 Accuracy   | INT8 Accuracy   |
+=======================+==========+============+==================+==================+=========+=================+=================+
| ResNet50 V1           | ImageNet | 128        | 191.17           | 1384.4           | 7.24    | 77.21%/93.55%   | 76.08%/93.04%   |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| MobileNet 1.0         | ImageNet | 128        | 565.21           | 3956.45          | 7.00    | 73.28%/91.22%   | 71.94%/90.47%   |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-VGG 300*          | VOC      | 224        | 19.05            | 113.62           | 5.96    | 77.4            | 77.38           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-VGG 512*          | VOC      | 224        | 6.78             | 37.62            | 5.55    | 78.41           | 78.38           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-resnet50_v1 512*  | VOC      | 224        | 28.59            | 143.7            | 5.03    | 80.21           | 80.25           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+
| SSD-mobilenet1.0 512* | VOC      | 224        | 65.97            | 212.59           | 3.22    | 75.42           | 74.70           |
+-----------------------+----------+------------+------------------+------------------+---------+-----------------+-----------------+

Quantized SSD models are evaluated with ``nms_thresh=0.45``, ``nms_topk=200``.

Demo usage for SSD
------------------

.. code:: bash

   # set omp to use all physical cores of one socket
   export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
   export CPUs=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
   export OMP_NUM_THREADS=$(CPUs)
   # with Pascal VOC validation dataset saved on disk
   python eval_ssd.py --network=vgg16_atrous --quantized --data-shape=300 --batch-size=224 --dataset=voc --benchmark

Usage:

::

   SYNOPSIS
            python eval_ssd.py [-h] [--network NETWORK] [--quantized]
                               [--data-shape DATA_SHAPE] [--batch-size BATCH_SIZE]
                               [--benchmark BENCHMARK] [--num-iterations NUM_ITERATIONS]
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
            --benchmark BENCHMARK  run dummy-data based benchmarking
            --num-iterations NUM_ITERATIONS  number of benchmarking iterations.
            --dataset DATASET     eval dataset.
            --num-workers NUM_WORKERS, -j NUM_WORKERS
                                    Number of data workers
            --num-gpus NUM_GPUS   number of gpus to use.
            --pretrained PRETRAINED
                                    Load weights from previously saved parameters.
            --save-prefix SAVE_PREFIX
                                    Saving parameter prefix
"""
