.. _gluoncv-model-zoo-action_recognition:

Action Recognition
==================

.. role:: greytag

Here is the model zoo for video action recognition task. We first show a visualization in the graph below, describing the inference throughputs vs. validation accuracy of Kinetics400 pre-trained models.

.. raw:: html
   :file: ../_static/ar_throughputs.html


.. hint::

  Training commands work with this script:
  :download:`Download train_recognizer.py<../../scripts/action-recognition/train_recognizer.py>`

  A model can have differently trained parameters with different hashtags.
  Parameters with :greytag:`a grey name` can be downloaded by passing the corresponding hashtag.

  - Download default pretrained weights: ``net = get_model('i3d_resnet50_v1_kinetics400', pretrained=True)``

  - Download weights given a hashtag: ``net = get_model('i3d_resnet50_v1_kinetics400', pretrained='568a722e')``

  The test script :download:`Download test_recognizer.py<../../scripts/action-recognition/test_recognizer.py>` can be used for
  evaluating the models on various datasets.

  The inference script :download:`Download inference.py<../../scripts/action-recognition/inference.py>` can be used for
  inferencing on a list of videos (demo purpose).

.. role:: tsntag


Kinetics400 Dataset
-------------------

The following table lists pre-trained models trained on Kinetics400.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  All models are trained using input size 224x224, except ``InceptionV3`` is trained and evaluated with input size of 299x299, ``C3D`` and ``R2+1D`` models are trained and evaluated with input size of 112x112.

  ``Clip Length`` is the number of frames within an input clip. ``32 (64/2)`` means we use 32 frames, but actually the frames are formed by randomly selecting 64 consecutive frames from the video and then skipping every other frame. This strategy is widely adopted to reduce computation and memory cost.

  ``Segments`` is the number of segments used during training. For testing (reporting these numbers), we use 250 views for 2D networks (25 frames and 10-crop) and 30 views for 3D networks (10 clips and 3-crop) following the convention.

  For ``SlowFast`` family of networks, our performance has a small gap to the numbers reported in the paper. This is because the official SlowFast implementation forces re-encoding every video to a fixed frame rate of 30. For fair comparison to other methods, we do not adopt that strategy, which leads to the small gap.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                              | Train Log                                                                                                                                          |
    +=============================================+==================+==============+================+===========+===========+============================================================================================================================================================+====================================================================================================================================================+
    | inceptionv1_kinetics400 [3]_                |   ImageNet       |      7       |       1        | 69.1      | 6dcdafb1  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv1_kinetics400_tsn.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv1_kinetics400_tsn.log>`_        |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | inceptionv3_kinetics400 [3]_                |   ImageNet       |      7       |       1        | 72.5      | 8a4a6946  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv3_kinetics400_tsn.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv3_kinetics400_tsn.log>`_        |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet18_v1b_kinetics400 [3]_               |   ImageNet       |      7       |       1        | 65.5      | 46d5a985  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet18_v1b_kinetics400_tsn.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet18_v1b_kinetics400_tsn.log>`_       |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet34_v1b_kinetics400 [3]_               |   ImageNet       |      7       |       1        | 69.1      | 8a8d0d8d  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet34_v1b_kinetics400_tsn.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet34_v1b_kinetics400_tsn.log>`_       |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet50_v1b_kinetics400 [3]_               |   ImageNet       |      7       |       1        | 69.9      | cc757e5c  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet50_v1b_kinetics400_tsn.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet50_v1b_kinetics400_tsn.log>`_       |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet101_v1b_kinetics400 [3]_              |   ImageNet       |      7       |       1        | 71.3      | 5bb6098e  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet101_v1b_kinetics400_tsn.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet101_v1b_kinetics400_tsn.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet152_v1b_kinetics400 [3]_              |   ImageNet       |      7       |       1        | 71.5      | 9bc70c66  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet152_v1b_kinetics400_tsn.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet152_v1b_kinetics400_tsn.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | c3d_kinetics400 [2]_                        |   Scratch        |      1       |    16 (32/2)   | 59.5      | a007b5fa  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/c3d_kinetics400.sh>`_                    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/c3d_kinetics400.log>`_                    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | p3d_resnet50_kinetics400 [5]_               |   Scratch        |      1       |    16 (32/2)   | 71.6      | 671ba81c  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/p3d_resnet50_kinetics400.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/p3d_resnet50_kinetics400.log>`_           |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | p3d_resnet101_kinetics400 [5]_              |   Scratch        |      1       |    16 (32/2)   | 72.6      | b30e3a63  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/p3d_resnet101_kinetics400.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/p3d_resnet101_kinetics400.log>`_          |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | r2plus1d_resnet18_kinetics400 [6]_          |   Scratch        |      1       |    16 (32/2)   | 70.8      | 5a14d1f9  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/r2plus1d_resnet18_kinetics400.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/r2plus1d_resnet18_kinetics400.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | r2plus1d_resnet34_kinetics400 [6]_          |   Scratch        |      1       |    16 (32/2)   | 71.6      | de2e592b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/r2plus1d_resnet34_kinetics400.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/r2plus1d_resnet34_kinetics400.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | r2plus1d_resnet50_kinetics400 [6]_          |   Scratch        |      1       |    16 (32/2)   | 73.9      | deaefb14  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/r2plus1d_resnet50_kinetics400.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/r2plus1d_resnet50_kinetics400.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_inceptionv1_kinetics400 [4]_            |   ImageNet       |      1       |    32 (64/2)   | 71.8      | 81e0be10  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv1_kinetics400.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv1_kinetics400.log>`_        |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_inceptionv3_kinetics400 [4]_            |   ImageNet       |      1       |    32 (64/2)   | 73.6      | f14f8a99  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv3_kinetics400.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv3_kinetics400.log>`_        |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_kinetics400 [4]_            |   ImageNet       |      1       |    32 (64/2)   | 74.0      | 568a722e  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet50_v1_kinetics400.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet50_v1_kinetics400.log>`_        |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet101_v1_kinetics400 [4]_           |   ImageNet       |      1       |    32 (64/2)   | 75.1      | 6b69f655  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet101_v1_kinetics400.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet101_v1_kinetics400.log>`_       |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl5_resnet50_v1_kinetics400 [7]_        |   ImageNet       |      1       |    32 (64/2)   | 75.2      | 3c0e47ea  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet50_v1_kinetics400.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet50_v1_kinetics400.log>`_    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl10_resnet50_v1_kinetics400 [7]_       |   ImageNet       |      1       |    32 (64/2)   | 75.3      | bfb58c41  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet50_v1_kinetics400.sh>`_   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet50_v1_kinetics400.log>`_   |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl5_resnet101_v1_kinetics400 [7]_       |   ImageNet       |      1       |    32 (64/2)   | 76.0      | fbfc1d30  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet101_v1_kinetics400.sh>`_   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet101_v1_kinetics400.log>`_   |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl10_resnet101_v1_kinetics400 [7]_      |   ImageNet       |      1       |    32 (64/2)   | 76.1      | 59186c31  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet101_v1_kinetics400.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet101_v1_kinetics400.log>`_  |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | slowfast_4x16_resnet50_kinetics400 [8]_     |   Scratch        |      1       |    36 (64/1)   | 75.3      | 9d650f51  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/slowfast_4x16_resnet50_kinetics400.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/slowfast_4x16_resnet50_kinetics400.log>`_ |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | slowfast_8x8_resnet50_kinetics400 [8]_      |   Scratch        |      1       |    40 (64/1)   | 76.6      | d6b25339  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/slowfast_8x8_resnet50_kinetics400.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/slowfast_8x8_resnet50_kinetics400.log>`_  |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | slowfast_8x8_resnet101_kinetics400 [8]_     |   Scratch        |      1       |    40 (64/1)   | 77.2      | fbde1a7c  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/slowfast_8x8_resnet101_kinetics400.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/slowfast_8x8_resnet101_kinetics400.log>`_ |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+


UCF101 Dataset
--------------

The following table lists pre-trained models trained on UCF101.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  The top-1 accuracy number shown below is for official split 1 of UCF101 dataset, not the average of 3 splits.

  ``InceptionV3`` is trained and evaluated with input size of 299x299.

  ``K400`` is Kinetics400 dataset, which means we use model pretrained on Kinetics400 as weights initialization.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                            | Train Log                                                                                                                                        |
    +=============================================+==================+==============+================+===========+===========+==========================================================================================================================================================+==================================================================================================================================================+
    | vgg16_ucf101 [3]_                           |   ImageNet       |      3       |       1        | 83.4      | d6dc1bba  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101_tsn.sh>`_                      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101_tsn.log>`_                      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | vgg16_ucf101 [1]_                           |   ImageNet       |      1       |       1        | 81.5      | 05e319d4  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101.sh>`_                          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101.log>`_                          |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | inceptionv3_ucf101 [3]_                     |   ImageNet       |      3       |       1        | 88.1      | 13ef5c3b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101_tsn.sh>`_                | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101_tsn.log>`_                |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | inceptionv3_ucf101 [1]_                     |   ImageNet       |      1       |       1        | 85.6      | 0c453da8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101.sh>`_                    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101.log>`_                    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_ucf101 [4]_                 |   ImageNet       |      1       |  32 (64/2)     | 83.9      | 7afc7286  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101.sh>`_                | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101.log>`_                |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_ucf101 [4]_                 | ImageNet, K400   |      1       |  32 (64/2)     | 95.4      | 760d0981  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101_kinetics400.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101_kinetics400.log>`_    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+


HMDB51 Dataset
--------------

The following table lists pre-trained models trained on HMDB51.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  The top-1 accuracy number shown below is for official split 1 of HMDB51 dataset, not the average of 3 splits.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                            | Train Log                                                                                                                                        |
    +=============================================+==================+==============+================+===========+===========+==========================================================================================================================================================+==================================================================================================================================================+
    | resnet50_v1b_hmdb51 [3]_                    |   ImageNet       |      3       |       1        | 55.2      | 682591e2  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51_tsn.sh>`_               | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51_tsn.log>`_               |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet50_v1b_hmdb51 [1]_                    |   ImageNet       |      1       |       1        | 52.2      | ba66ee4b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51.sh>`_                   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51.log>`_                   |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_hmdb51 [4]_                 |   ImageNet       |      1       |  32 (64/2)     | 48.5      | 0d0ad559  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51.sh>`_                | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51.log>`_                |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_hmdb51 [4]_                 | ImageNet, K400   |      1       |  32 (64/2)     | 70.9      | 2ec6bf01  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51_kinetics400.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51_kinetics400.log>`_    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+



Something-Something-V2 Dataset
------------------------------

The following table lists pre-trained models trained on Something-Something-V2.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.


.. table::
    :widths: 40 8 8 8 10 8 8 10

    +--------------------------------------+------------------+--------------+----------------+-----------+-----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                 |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                                     | Train Log                                                                                                                                               |
    +======================================+==================+==============+================+===========+===========+===================================================================================================================================================================+=========================================================================================================================================================+
    | resnet50_v1b_sthsthv2 [3]_           |   ImageNet       |      8       |       1        | 35.5      | 80ee0c6b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/resnet50_v1b_sthsthv2_tsn.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/resnet50_v1b_sthsthv2_tsn.log>`_      |
    +--------------------------------------+------------------+--------------+----------------+-----------+-----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_sthsthv2 [4]_        |   ImageNet       |      1       |    16 (32/2)   | 50.6      | 01961e4c  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/i3d_resnet50_v1_sthsthv2.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/i3d_resnet50_v1_sthsthv2.log>`_       |
    +--------------------------------------+------------------+--------------+----------------+-----------+-----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+


.. [1] Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao. \
       "Towards Good Practices for Very Deep Two-Stream ConvNets." \
       arXiv preprint arXiv:1507.02159, 2015.
.. [2] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani and Manohar Paluri. \
       "Learning Spatiotemporal Features with 3D Convolutional Networks." \
       In International Conference on Computer Vision (ICCV), 2015.
.. [3] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang and Luc Van Gool. \
       "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition." \
       In European Conference on Computer Vision (ECCV), 2016.
.. [4] Joao Carreira and Andrew Zisserman. \
       "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset." \
       In Computer Vision and Pattern Recognition (CVPR), 2017.
.. [5] Zhaofan Qiu, Ting Yao and Tao Mei. \
       "SLearning Spatio-Temporal Representation with Pseudo-3D Residual Networks." \
       In International Conference on Computer Vision (ICCV), 2017.
.. [6] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun and Manohar Paluri. \
       "A Closer Look at Spatiotemporal Convolutions for Action Recognition." \
       In Computer Vision and Pattern Recognition (CVPR), 2018.
.. [7] Xiaolong Wang, Ross Girshick, Abhinav Gupta and Kaiming He. \
       "Non-local Neural Networks." \
       In Computer Vision and Pattern Recognition (CVPR), 2018.
.. [8] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik and Kaiming He. \
       "SlowFast Networks for Video Recognition." \
       In International Conference on Computer Vision (ICCV), 2019.
