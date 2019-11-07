.. _gluoncv-model-zoo-action_recognition:

Action Recognition
==================

.. role:: greytag

Table of pre-trained models for video action recognition and their performance.

.. hint::

  Training commands work with this script:
  :download:`Download train_recognizer.py<../../scripts/action-recognition/train_recognizer.py>`

  A model can have differently trained parameters with different hashtags.
  Parameters with :greytag:`a grey name` can be downloaded by passing the corresponding hashtag.

  - Download default pretrained weights: ``net = get_model('inceptionv3_ucf101', pretrained=True)``

  - Download weights given a hashtag: ``net = get_model('inceptionv3_ucf101', pretrained='0c453da8')``

  The test script :download:`Download test_recognizer.py<../../scripts/action-recognition/test_recognizer.py>` can be used for
  evaluating the models.

.. role:: tsntag

UCF101 Dataset
--------------

The following table lists pre-trained models trained on UCF101.

.. note::

  Our pre-trained models reproduce results from "Temporal Segment Networks" [2]_ and "Inflated 3D Networks (I3D)" [3]_ . Please check the reference paper for further information.

  The top-1 accuracy number shown below is for official split 1 of UCF101 dataset, not the average of 3 splits.

  ``InceptionV3`` is trained and evaluated with input size of 299x299.

  ``K400`` is Kinetics400 dataset, which means we use model pretrained on Kinetics400 as weights initialization.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                            | Train Log                                                                                                                                        |
    +=============================================+==================+==============+================+===========+===========+==========================================================================================================================================================+==================================================================================================================================================+
    | vgg16_ucf101 [2]_                           |   ImageNet       |      3       |       1        | 83.4      | d6dc1bba  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101_tsn.sh>`_                      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101_tsn.log>`_                      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | vgg16_ucf101 [1]_                           |   ImageNet       |      1       |       1        | 81.5      | 05e319d4  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101.sh>`_                          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/vgg16_ucf101.log>`_                          |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | inceptionv3_ucf101 [2]_                     |   ImageNet       |      3       |       1        | 88.1      | 13ef5c3b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101_tsn.sh>`_                | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101_tsn.log>`_                |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | inceptionv3_ucf101 [1]_                     |   ImageNet       |      1       |       1        | 85.6      | 0c453da8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101.sh>`_                    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/inceptionv3_ucf101.log>`_                    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_ucf101 [3]_                 |   ImageNet       |      1       |  32 (64/2)     | 83.9      | 7afc7286  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101.sh>`_                | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101.log>`_                |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_ucf101 [3]_                 | ImageNet, K400   |      1       |  32 (64/2)     | 95.4      | 760d0981  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101_kinetics400.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/ucf101/i3d_resnet50_v1_ucf101_kinetics400.log>`_    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+


HMDB51 Dataset
--------------

The following table lists pre-trained models trained on HMDB51.

.. note::

  Our pre-trained models reproduce results from "Temporal Segment Networks" [2]_ and "Inflated 3D Networks (I3D)" [3]_ . Please check the reference paper for further information.

  The top-1 accuracy number shown below is for official split 1 of HMDB51 dataset, not the average of 3 splits.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                            | Train Log                                                                                                                                        |
    +=============================================+==================+==============+================+===========+===========+==========================================================================================================================================================+==================================================================================================================================================+
    | resnet50_v1b_hmdb51 [2]_                    |   ImageNet       |      3       |       1        | 55.2      | 682591e2  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51_tsn.sh>`_               | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51_tsn.log>`_               |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet50_v1b_hmdb51 [1]_                    |   ImageNet       |      1       |       1        | 52.2      | ba66ee4b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51.sh>`_                   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/resnet50_v1b_hmdb51.log>`_                   |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_hmdb51 [3]_                 |   ImageNet       |      1       |  32 (64/2)     | 48.5      | 0d0ad559  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51.sh>`_                | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51.log>`_                |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_hmdb51 [3]_                 | ImageNet, K400   |      1       |  32 (64/2)     | 70.9      | 2ec6bf01  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51_kinetics400.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/hmdb51/i3d_resnet50_v1_hmdb51_kinetics400.log>`_    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+


Kinetics400 Dataset
-------------------

The following table lists pre-trained models trained on Kinetics400.

.. note::

  Our pre-trained models reproduce results from "Temporal Segment Networks (TSN)" [2]_ , "Inflated 3D Networks (I3D)" [3]_ , "Non-local Neural Networks" [4]_ . Please check the reference paper for further information.

  ``InceptionV3`` is trained and evaluated with input size of 299x299.

  ``Clip Length`` is the number of frames within an input clip. ``32 (64/2)`` means we use 32 frames, but actually the frames are formed by randomly selecting 64 consecutive frames from the video and then skipping every other frame. This strategy is widely adopted to reduce computation and memory cost.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                        |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                            | Train Log                                                                                                                                        |
    +=============================================+==================+==============+================+===========+===========+==========================================================================================================================================================+==================================================================================================================================================+
    | inceptionv3_kinetics400 [2]_                |   ImageNet       |      3       |       1        | 72.5      | 8a4a6946  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv3_kinetics400_tsn.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/inceptionv3_kinetics400_tsn.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet18_v1b_kinetics400 [2]_               |   ImageNet       |      7       |       1        | 66.4      | 9d5cf9ec  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet18_v1b_kinetics400_tsn.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet18_v1b_kinetics400_tsn.log>`_     |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet34_v1b_kinetics400 [2]_               |   ImageNet       |      7       |       1        | 69.5      | b91fcb2f  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet34_v1b_kinetics400_tsn.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet34_v1b_kinetics400_tsn.log>`_     |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet50_v1b_kinetics400 [2]_               |   ImageNet       |      7       |       1        | 70.6      | e3ad0758  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet50_v1b_kinetics400_tsn.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet50_v1b_kinetics400_tsn.log>`_     |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet101_v1b_kinetics400 [2]_              |   ImageNet       |      7       |       1        | 71.5      | f0a8dcb0  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet101_v1b_kinetics400_tsn.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet101_v1b_kinetics400_tsn.log>`_    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | resnet152_v1b_kinetics400 [2]_              |   ImageNet       |      7       |       1        | 72.3      | 1968220d  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet152_v1b_kinetics400_tsn.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/resnet152_v1b_kinetics400_tsn.log>`_    |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_inceptionv1_kinetics400 [3]_            |   ImageNet       |      1       |    32 (64/2)   | 71.7      | f36bdeed  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv1_kinetics400.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv1_kinetics400.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_inceptionv3_kinetics400 [3]_            |   ImageNet       |      1       |    32 (64/2)   | 73.3      | bbd4185a  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv3_kinetics400.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_inceptionv3_kinetics400.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_kinetics400 [4]_            |   ImageNet       |      1       |    32 (64/2)   | 73.6      | 254ae7d9  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet50_v1_kinetics400.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet50_v1_kinetics400.log>`_      |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet101_v1_kinetics400 [4]_           |   ImageNet       |      1       |    32 (64/2)   | 74.8      | c5721407  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet101_v1_kinetics400.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_resnet101_v1_kinetics400.log>`_     |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl5_resnet50_v1_kinetics400 [4]_        |   ImageNet       |      1       |    32 (64/2)   | 73.9      | 382433ba  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet50_v1_kinetics400.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet50_v1_kinetics400.log>`_  |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl10_resnet50_v1_kinetics400 [4]_       |   ImageNet       |      1       |    32 (64/2)   | 74.5      | 26b41dd6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet50_v1_kinetics400.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet50_v1_kinetics400.log>`_ |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl5_resnet101_v1_kinetics400 [4]_       |   ImageNet       |      1       |    32 (64/2)   | 75.2      | 8b25d02f  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet101_v1_kinetics400.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl5_resnet101_v1_kinetics400.log>`_ |
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_nl10_resnet101_v1_kinetics400 [4]_      |   ImageNet       |      1       |    32 (64/2)   | 75.3      | 77d7ed77  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet101_v1_kinetics400.sh>`_| `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/kinetics400/i3d_nl10_resnet101_v1_kinetics400.log>`_|
    +---------------------------------------------+------------------+--------------+----------------+-----------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------+

Something-Something-V2 Dataset
------------------------------

The following table lists pre-trained models trained on Something-Something-V2.

.. note::

  Our pre-trained models reproduce results from "Temporal Segment Networks (TSN)" [2]_ , "Inflated 3D Networks (I3D)" [3]_ . Please check the reference paper for further information.


.. table::
    :widths: 40 8 8 8 10 8 8 10

    +--------------------------------------+------------------+--------------+----------------+-----------+-----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                 |   Pretrained     |    Segments  |   Clip Length  | Top-1     | Hashtag   | Train Command                                                                                                                                                     | Train Log                                                                                                                                               |
    +======================================+==================+==============+================+===========+===========+===================================================================================================================================================================+=========================================================================================================================================================+
    | resnet50_v1b_sthsthv2 [2]_           |   ImageNet       |      8       |       1        | 35.5      | 80ee0c6b  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/resnet50_v1b_sthsthv2_tsn.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/resnet50_v1b_sthsthv2_tsn.log>`_      |
    +--------------------------------------+------------------+--------------+----------------+-----------+-----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
    | i3d_resnet50_v1_sthsthv2 [3]_        |   ImageNet       |      1       |    16 (32/2)   | 50.6      | 01961e4c  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/i3d_resnet50_v1_sthsthv2.sh>`_         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/action_recognition/somethingsomethingv2/i3d_resnet50_v1_sthsthv2.log>`_       |
    +--------------------------------------+------------------+--------------+----------------+-----------+-----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+


.. [1] Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao. \
       "Towards Good Practices for Very Deep Two-Stream ConvNets." \
       arXiv preprint arXiv:1507.02159, 2015.
.. [2] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang and Luc Van Gool. \
       "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition." \
       In European Conference on Computer Vision (ECCV), 2016.
.. [3] Joao Carreira and Andrew Zisserman. \
       "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset." \
       In Computer Vision and Pattern Recognition (CVPR), 2017.
.. [4] Xiaolong Wang, Ross Girshick, Abhinav Gupta and Kaiming He. \
       "Non-local Neural Networks." \
       In Computer Vision and Pattern Recognition (CVPR), 2018.

