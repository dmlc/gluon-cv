.. _gluoncv-model-zoo-depth:

Depth Prediction
================


Here is the model zoo for task of depth prediction.


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

    python test.py --model_zoo monodepth2_resnet18_kitti_stereo_640x192 --pretrained_type gluoncv --eval_stereo --png



.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  ``Modality`` is the method used during training. ``Stereo`` means we use stereo image pairs to calculate the loss.

  ``Resolution`` is the input size of the model during training. ``640x192`` means we resize the raw image (1242x375) to 640x192.

.. table::
    :widths: 40 8 8 8 10 8 8 10

    +-------------------------------------------------------+------------------+--------------+-----------------+--------------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
    | Name                                                  |   Modality       |   Resolution | Abs. Rel. Error | delta < 1.25 | Hashtag   | Train Command                                                                                                                                              | Train Log                                                                                                                                          |
    +=======================================================+==================+==============+=================+==============+===========+============================================================================================================================================================+====================================================================================================================================================+
    | monodepth2_resnet18_kitti_stereo_640x192 [1]_         |   Stereo         |  640x192     |     0.114       | 0.856        | 92871317  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/depth/kitti/monodepth2_resnet18_kitti_stereo_640x192.sh>`_              | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/depth/kitti/monodepth2_resnet18_kitti_stereo_640x192.log>`_              |
    +-------------------------------------------------------+------------------+--------------+-----------------+--------------+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+



.. [1] Clement Godard, Oisin Mac Aodha, Michael Firman and Gabriel J. Brostow. \
       "Digging into Self-Supervised Monocular Depth Prediction." \
       Proceedings of the International Conference on Computer Vision (ICCV), 2019.