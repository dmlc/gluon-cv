Here is the PyTorch model zoo for video action recognition task.

.. hint::

  Training commands work with this script:
  :download:`Download train_ddp_pytorch.py<../../scripts/action-recognition/train_ddp_pytorch.py>`

  ``python train_ddp_pytorch.py --config-file CONFIG``

  The test script :download:`Download test_ddp_pytorch.py<../../scripts/action-recognition/test_ddp_pytorch.py>` can be used for
  performance evaluation on various datasets. Please set ``MODEL.PRETRAINED = True`` in the configuration file if you would like to use
  the trained models in our model zoo.

  ``python test_ddp_pytorch.py --config-file CONFIG``


Kinetics400 Dataset
-------------------

The following table lists our trained models on Kinetics400.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  All models are trained using input size 224x224, except ``R2+1D`` models are trained and evaluated with input size of 112x112.

  ``Clip Length`` is the number of frames within an input clip. ``32 (64/2)`` means we use 32 frames, but actually the frames are formed by randomly selecting 64 consecutive frames from the video and then skipping every other frame. This strategy is widely adopted to reduce computation and memory cost.

  ``Segment`` is the number of segments used during training. For testing (reporting these numbers), we use 250 views for 2D networks (25 frames and 10-crop) and 30 views for 3D networks (10 clips and 3-crop) following the convention.

  The model weights of ``r2plus1d_v2_resnet152_kinetics400``, ``ircsn_v2_resnet152_f32s2_kinetics400`` and ``TPN family`` are ported from VMZ and TPN repository. You may ignore the training config of these models for now.


.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Kinetics400_torch.csv
   :header-rows: 1
   :class: tight-table
   :widths: 36 12 10 10 8 12 12


Kinetics700 Dataset
-------------------

The following table lists our trained models on Kinetics700.

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Kinetics700_torch.csv
   :header-rows: 1
   :class: tight-table
   :widths: 36 12 10 10 8 12 12


Something-Something-V2 Dataset
------------------------------

The following table lists our trained models on Something-Something-V2.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Something-Something-V2_torch.csv
   :header-rows: 1
   :class: tight-table
   :widths: 36 12 10 10 8 12 12
