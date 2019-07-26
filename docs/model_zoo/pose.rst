.. _gluoncv-model-zoo-classification:

Pose Estimation
====================

.. note::

    Pose Estimation is released in GluonCV 0.4. Please be sure to update your installation by
    ``pip install gluoncv --upgrade`` to try it out.

MS COCO Keypoints
~~~~~~~~

.. hint::

  The training commands work with the following scripts:

  - For Simple Pose [1]_ networks: :download:`Download train_simple_pose.py<../../scripts/pose/simple_pose/train_simple_pose.py>`

.. hint::

    For COCO dataset, training imageset is train2017 and validation imageset is val2017.

    The COCO metric, Average Precision (AP) with IoU threshold 0.5:0.95 (averaged 10 values, AP 0.5:0.95), 0.5 (AP 0.5) and 0.75 (AP 0.75) are reported together in the format (AP 0.5:0.95)/(AP 0.5)/(AP 0.75).

    COCO keypoints metrics evaluate Object Keypoint Similarity AP. Please read the `official doc <http://cocodataset.org/#keypoints-eval>`__ for detailed introduction.

    By averaging the prediction from the original input and the flipped one, we can get higher performance. Here we report the performance for predictions with and without the flip ensemble.

.. role:: tag

Simple Pose with ResNet
------

Checkout the demo tutorial here: :ref:`sphx_glr_build_examples_pose_demo_simple_pose.py`

Most models are trained with input size 256x192, unless specified.
Parameters with :greytag:`a grey name` can be downloaded by passing the corresponding hashtag.

- Download default pretrained weights: ``net = get_model('simple_pose_resnet152_v1d', pretrained=True)``
- Download weights given a hashtag: ``net = get_model('simple_pose_resnet152_v1d', pretrained='2f544338')``

.. table::
   :widths: 45 5 5 10 20 15

   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | Model                                            | OKS AP         | OKS AP (with flip) | Hashtag  | Training Command                                                                                                                      | Training log                                                                                                                  |
   +==================================================+================+====================+==========+=======================================================================================================================================+===============================================================================================================================+
   | simple_pose_resnet18_v1b [1]_                    | 66.3/89.2/73.4 | 68.4/90.3/75.7     | f63d42ac | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet18_v1b_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet18_v1b_coco.log>`_           |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet18_v1b [1]_ :gray:`(128x96)`   | 52.8/83.6/57.9 | 54.5/84.8/60.3     | ccd24037 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet18_v1b_small_coco.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet18_v1b_small_coco.log>`_     |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet50_v1b [1]_                    | 71.0/91.2/78.6 | 72.2/92.2/79.9     | e2c7b1ad | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet50_v1b_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet50_v1b_coco.log>`_           |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet50_v1d [1]_                    | 71.6/91.3/78.7 | 73.3/92.4/80.8     | ba2675b6 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet50_v1d_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet50_v1d_coco.log>`_           |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet101_v1b [1]_                   | 72.4/92.2/79.8 | 73.7/92.3/81.1     | b7ec0de1 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet101_v1b_coco.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet101_v1b_coco.log>`_          |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet101_v1d [1]_                   | 73.0/92.2/80.8 | 74.2/92.4/82.0     | 1f8f48fd | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet101_v1d_coco.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet101_v1d_coco.log>`_          |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet152_v1b [1]_                   | 72.4/92.1/79.6 | 74.2/92.3/82.1     | ef4e0336 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet152_v1b_coco.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet152_v1b_coco.log>`_          |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet152_v1d [1]_                   | 73.4/92.3/80.7 | 74.6/93.4/82.1     | 3ca502ea | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet152_v1d_coco.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet152_v1d_coco.log>`_          |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
   | simple_pose_resnet152_v1d [1]_ :gray:`(384x288)` | 74.8/92.3/82.0 | 76.1/92.4/83.2     | 2f544338 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet152_v1d_large_coco.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/pose/simple_pose_resnet152_v1d_large_coco.log>`_    |
   +--------------------------------------------------+----------------+--------------------+----------+---------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

.. [1] Xiao, Bin, Haiping Wu, and Yichen Wei. \
       "Simple baselines for human pose estimation and tracking." \
       Proceedings of the European Conference on Computer Vision (ECCV). 2018.
