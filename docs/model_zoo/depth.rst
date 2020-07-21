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