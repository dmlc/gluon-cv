Here is the model zoo for the task of depth prediction.


.. hint::

  Training commands work with this script:
  :download:`Download train.py<../../scripts/depth/train.py>`

  The test script :download:`Download test.py<../../scripts/depth/test.py>` can be used for
  evaluating the models on various datasets.


KITTI Dataset
-------------------

The following table lists pre-trained models trained on KITTI.

.. hint::

  The test script :download:`Download test.py<../../scripts/depth/test.py>` can be used for
  evaluating the models (KITTI RAW results are evaluated using the official server). For example
  ``monodepth2_resnet18_kitti_stereo_640x192``::

    python test.py --model_zoo monodepth2_resnet18_kitti_stereo_640x192 --pretrained_type gluoncv --batch_size 1 --eval_stereo --png


.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  ``Modality`` is the method used during training. ``Stereo`` means we use stereo image pairs to calculate the loss,  ``Mono`` means we use monocular image sequences to calculate the loss,
  ``Mono + Stereo`` means both the stereo image pairs and monocular image sequences are used to calculate the loss.

  ``Resolution`` is the input size of the model during training. ``640x192`` means we resize the raw image (1242x375) to 640x192.

.. csv-table::
   :file: ./csv_tables/Depths/KITTI.csv
   :header-rows: 1
   :class: tight-table
   :widths: 37 9 10 8 8 10 10 8

PoseNet
-------------------

Monodepth2 trains depth and pose models at the same time via a self-supervised manner. So, we also give reproduced results of our pre-trained models here.

.. hint::

  The test script :download:`Download test_pose.py<../../scripts/depth/test_pose.py>` can be used for
  evaluating the models (KITTI Odometry results are evaluated using the official server). For example
  ``monodepth2_resnet18_posenet_kitti_mono_stereo_640x192``::

    python test_pose.py --model_zoo_pose monodepth2_resnet18_posenet_kitti_mono_640x192 --data_path ~/.mxnet/datasets/kitti/kitti_odom --eval_split odom_9  --pretrained_type gluoncv --batch_size 1 --png

  Please check the full tutorials `Testing PoseNet from image sequences with pre-trained Monodepth2 Pose models <../build/examples_depth/test_monodepth2_posenet.html>`_.


.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  ``Sequence 09`` and ``Sequence 10`` means the model is tested on sequence 9 and sequence 10 of the KITTI Odometry dataset respectively.
  Results show the average absolute trajectory error (ATE), and standard deviation, in meter.

.. csv-table::
   :file: ./csv_tables/Depths/PoseNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 40 10 10 13 13

