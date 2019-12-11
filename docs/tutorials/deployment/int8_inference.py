"""3. Inference with Quantized Models
=====================================

This is a tutorial which illustrates how to use quantized GluonCV
models for inference on Intel Xeon Processors to gain higher performance.

The following example requires ``GluonCV>=0.5`` and ``MXNet-mkl>=1.6.0b20191010``. Please follow `our installation guide <../../index.html#installation>`__ to install or upgrade GluonCV and nightly build of MXNet if necessary.

Introduction
------------

GluonCV delivered some quantized models to improve the performance and reduce the deployment costs for the computer vision inference tasks. In real production, there are two main benefits of lower precision (INT8). First, the computation can be accelerated by the low precision instruction, like Intel Vector Neural Network Instruction (VNNI). Second, lower precision data type would save the memory bandwidth and allow for better cache locality and save the power. The new feature can get up to 4X performance speedup in the latest `AWS EC2 C5 instances <https://aws.amazon.com/blogs/aws/now-available-new-c5-instance-sizes-and-bare-metal-instances/>`_ under the `Intel Deep Learning Boost (VNNI) <https://www.intel.ai/intel-deep-learning-boost/>`_ enabled hardware with less than 0.5% accuracy drop.

Please checkout `verify_pretrained.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/classification/imagenet/verify_pretrained.py>`_ for imagenet inference,
`eval_ssd.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/detection/ssd/eval_ssd.py>`_ for SSD inference, `test.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/segmentation/test.py>`_ 
for segmentation inference, `validate.py <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/pose/simple_pose/validate.py>`_ for pose estimation inference, and `test_recognizer.py <https://github.com/dmlc/gluon-cv/blob/master/scripts/action-recognition/test_recognizer.py>`_ for video action recognition.

Performance
-----------

GluonCV supports some quantized classification models, detection models and segmentation models.
For the throughput, the target is to achieve the maximum machine efficiency to combine the inference requests together and get the results by one iteration. From the bar-chart, it is clearly that the fusion and quantization approach improved the throughput from 2.68X to 7.24X for selected models.
Below CPU performance is collected with dummy input from Intel(R) VNNI enabled AWS EC2 C5.12xlarge instance with 24 physical cores.

.. figure:: https://user-images.githubusercontent.com/34727741/67351790-ecdc7280-f580-11e9-8b44-1b4548cb6031.png
   :alt: Gluon Quantization Performance


+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
|  Model                      | Dataset        | Batch Size | Speedup (INT8/FP32) | FP32 Accuracy   | INT8 Accuracy   |
+=============================+================+============+=====================+=================+=================+
| ResNet50 V1                 | ImageNet       | 128        | 7.24                | 77.21%/93.55%   | 76.08%/93.04%   |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| MobileNet 1.0               | ImageNet       | 128        | 7.00                | 73.28%/91.22%   | 71.94%/90.47%   |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| SSD-VGG 300*                | VOC            | 224        | 5.96                | 77.4            | 77.38           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| SSD-VGG 512*                | VOC            | 224        | 5.55                | 78.41           | 78.38           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| SSD-resnet50_v1 512*        | VOC            | 224        | 5.03                | 80.21           | 80.25           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| SSD-mobilenet1.0 512*       | VOC            | 224        | 3.22                | 75.42           | 74.70           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| FCN_resnet101               | VOC            | 1          | 4.82                | 97.97%          | 98.00%          |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| PSP_resnet101               | VOC            | 1          | 2.68                | 98.46%          | 98.45%          |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| Deeplab_resnet101           | VOC            | 1          | 3.20                | 98.36%          | 98.34%          |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| FCN_resnet101               | COCO           | 1          | 5.05                | 91.28%          | 90.96%          |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| PSP_resnet101               | COCO           | 1          | 2.69                | 91.82%          | 91.88%          |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| Deeplab_resnet101           | COCO           | 1          | 3.27                | 91.86%          | 91.98%          |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| simple_pose_resnet18_v1b    | COCO Keypoint  | 128        | 2.55                | 66.3            | 65.9            |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| simple_pose_resnet50_v1b    | COCO Keypoint  | 128        | 3.50                | 71.0            | 70.6            |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| simple_pose_resnet50_v1d    | COCO Keypoint  | 128        | 5.89                | 71.6            | 71.4            |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| simple_pose_resnet101_v1b   | COCO Keypoint  | 128        | 4.07                | 72.4            | 72.2            |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| simple_pose_resnet101_v1d   | COCO Keypoint  | 128        | 5.97                | 73.0            | 72.7            |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| vgg16_ucf101                | UCF101         | 64         | 4.46                | 81.86           | 81.41           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| inceptionv3_ucf101          | UCF101         | 64         | 5.16                | 86.92           | 86.55           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| resnet18_v1b_kinetics400    | Kinetics400    | 64         | 5.24                | 63.29           | 63.14           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| resnet50_v1b_kinetics400    | Kinetics400    | 64         | 6.78                | 68.08           | 68.15           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+
| inceptionv3_kinetics400     | Kinetics400    | 64         | 5.29                | 67.93           | 67.92           |
+-----------------------------+----------------+------------+---------------------+-----------------+-----------------+

Quantized SSD models are evaluated with ``nms_thresh=0.45``, ``nms_topk=200``. For segmentation models, the accuracy metric is pixAcc, and for pose-estimation models, the accuracy metric is OKS AP w/o flip.
Quantized 2D video action recognition models are calibrated with ``num-segments=3`` (7 is for resnet-based models).

Demo usage for SSD
------------------

.. code:: bash

   # set omp to use all physical cores of one socket
   export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
   export CPUs=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
   export OMP_NUM_THREADS=$(CPUs)
   # with Pascal VOC validation dataset saved on disk
   python eval_ssd.py --network=mobilenet1.0 --quantized --data-shape=512 --batch-size=224 --dataset=voc --benchmark

Usage:

::

   SYNOPSIS
            python eval_ssd.py [-h] [--network NETWORK] [--deploy]
                               [--model-prefix] [--quantized]
                               [--data-shape DATA_SHAPE] [--batch-size BATCH_SIZE]
                               [--benchmark BENCHMARK] [--num-iterations NUM_ITERATIONS]
                               [--dataset DATASET] [--num-workers NUM_WORKERS]
                               [--num-gpus NUM_GPUS] [--pretrained PRETRAINED]
                               [--save-prefix SAVE_PREFIX] [--calibration CALIBRATION]
                               [--num-calib-batches NUM_CALIB_BATCHES]
                               [--quantized-dtype {auto,int8,uint8}]
                               [--calib-mode CALIB_MODE]

   OPTIONS
            -h, --help              show this help message and exit
            --network NETWORK       base network name
            --deploy                whether load static model for deployment
            --model-prefix MODEL_PREFIX
                                    load static model as hybridblock.          
            --quantized             use int8 pretrained model
            --data-shape DATA_SHAPE
                                    input data shape
            --batch-size BATCH_SIZE
                                    eval mini-batch size
            --benchmark BENCHMARK   run dummy-data based benchmarking
            --num-iterations NUM_ITERATIONS  number of benchmarking iterations.
            --dataset DATASET    eval dataset.
            --num-workers NUM_WORKERS, -j NUM_WORKERS
                                    number of data workers
            --num-gpus NUM_GPUS     number of gpus to use.
            --pretrained PRETRAINED
                                    load weights from previously saved parameters.
            --save-prefix SAVE_PREFIX
                                    saving parameter prefix
            --calibration           quantize model
            --num-calib-batches NUM_CALIB_BATCHES
                                    number of batches for calibration
            --quantized-dtype {auto,int8,uint8}
                                    quantization destination data type for input data
            --calib-mode CALIB_MODE
                                    calibration mode used for generating calibration table
                                    for the quantized symbol; supports 1. none: no
                                    calibration will be used. The thresholds for
                                    quantization will be calculated on the fly. This will
                                    result in inference speed slowdown and loss of
                                    accuracy in general. 2. naive: simply take min and max
                                    values of layer outputs as thresholds for
                                    quantization. In general, the inference accuracy
                                    worsens with more examples used in calibration. It is
                                    recommended to use `entropy` mode as it produces more
                                    accurate inference results. 3. entropy: calculate KL
                                    divergence of the fp32 output and quantized output for
                                    optimal thresholds. This mode is expected to produce
                                    the best inference accuracy of all three kinds of
                                    quantized models if the calibration dataset is
                                    representative enough of the inference dataset.                                  

Calibration Tool
----------------

GluonCV also delivered calibration tool for users to quantize their models into int8 with their own dataset. Currently, calibration tool only supports hybridized gluon models. Below is an example of quantizing SSD model.

.. code:: bash

   # Calibration
   python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=224 --dataset=voc --calibration --num-calib-batches=5 --calib-mode=naive
   # INT8 Inference
   python eval_ssd.py --network=mobilenet1.0 --data-shape=512 --batch-size=224 --deploy --model-prefix=./model/ssd_512_mobilenet1.0_voc-quantized-naive

The first command will launch naive calibration to quantize your ssd_mobilenet1.0 model to int8 by using a subset (5 batches) of your given dataset. Users can tune the int8 accuracy by setting different calibration configurations. After calibration, quantized model and parameter will be saved on your disk. Then, the second command will load quantized model as a symbolblock for inference.

Users can also quantize their own gluon hybridized model by using `quantize_net` api. Below are some descriptions.

API:

::

   CODE

      from mxnet.contrib.quantization import *
      quantized_net = quantize_net(network, quantized_dtype='auto',
                                   exclude_layers=None, exclude_layers_match=None,
                                   calib_data=None, data_shapes=None,
                                   calib_mode='naive', num_calib_examples=None,
                                   ctx=mx.cpu(), logger=logging)

   Parameters

      network : Gluon HybridBlock
                  Defines the structure of a neural network for FP32 data types.
      quantized_dtype : str
                  The quantized destination type for input data. Currently support 'int8'
                  , 'uint8' and 'auto'.
                  'auto' means automatically select output type according to calibration result.
                  Default value is 'int8'.
      exclude_layers : list of strings
                  A list of strings representing the names of the symbols that users want to excluding
      exclude_layers_match : list of strings
                  A list of strings wildcard matching the names of the symbols that users want to excluding
                  from being quantized.
      calib_data : mx.io.DataIter or gluon.DataLoader
                  A iterable data loading object.
      data_shapes : list
                  List of DataDesc, required if calib_data is not provided
      calib_mode : str
                  If calib_mode='none', no calibration will be used and the thresholds for
                  requantization after the corresponding layers will be calculated at runtime by
                  calling min and max operators. The quantized models generated in this
                  mode are normally 10-20% slower than those with calibrations during inference.
                  If calib_mode='naive', the min and max values of the layer outputs from a calibration
                  dataset will be directly taken as the thresholds for quantization.
                  If calib_mode='entropy', the thresholds for quantization will be
                  derived such that the KL divergence between the distributions of FP32 layer outputs and
                  quantized layer outputs is minimized based upon the calibration dataset.
      calib_layer : function
                  Given a layer's output name in string, return True or False for deciding whether to
                  calibrate this layer. If yes, the statistics of the layer's output will be collected;
                  otherwise, no information of the layer's output will be collected. If not provided,
                  all the layers' outputs that need requantization will be collected.
      num_calib_examples : int or None
                  The maximum number of examples that user would like to use for calibration.
                  If not provided, the whole calibration dataset will be used.
      ctx : Context
                  Defines the device that users want to run forward propagation on the calibration
                  dataset for collecting layer output statistics. Currently, only supports single context.
                  Currently only support CPU with MKL-DNN backend.
      logger : Object
                  A logging object for printing information during the process of quantization.

   Returns

      network : Gluon SymbolBlock
                  Defines the structure of a neural network for INT8 data types.
  
"""
