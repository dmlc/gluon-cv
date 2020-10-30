.. _gluoncv-model-zoo-action_recognition:

Action Recognition
==================

.. role:: framework
   :class: framework
.. role:: select
   :class: selected framework

.. container:: Frameworks

  .. container:: framework-group

     :framework:`MXNet`
     :framework:`Pytorch`

.. rst-class:: MXNet

MXNet
*************

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

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Kinetics400.csv
   :header-rows: 1
   :class: tight-table
   :widths: 30 12 10 10 8 10 12 8

UCF101 Dataset
--------------

The following table lists pre-trained models trained on UCF101.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  The top-1 accuracy number shown below is for official split 1 of UCF101 dataset, not the average of 3 splits.

  ``InceptionV3`` is trained and evaluated with input size of 299x299.

  ``K400`` is Kinetics400 dataset, which means we use model pretrained on Kinetics400 as weights initialization.

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/UCF101.csv
   :header-rows: 1
   :class: tight-table
   :widths: 30 12 10 10 8 10 12 8

HMDB51 Dataset
--------------

The following table lists pre-trained models trained on HMDB51.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  The top-1 accuracy number shown below is for official split 1 of HMDB51 dataset, not the average of 3 splits.

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/HMDB51.csv
   :header-rows: 1
   :class: tight-table
   :widths: 30 12 10 10 8 10 12 8

Something-Something-V2 Dataset
------------------------------

The following table lists pre-trained models trained on Something-Something-V2.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Something-Something-V2.csv
   :header-rows: 1
   :class: tight-table
   :widths: 30 12 10 10 8 10 12 8

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
       "Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks." \
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


.. rst-class:: Pytorch

PyTorch
*************


Here is the PyTorch model zoo for video action recognition task.

.. hint::

  Training commands work with this script:
  :download:`Download train_ddp_pytorch.py<../../scripts/action-recognition/train_ddp_pytorch.py>`

  ``python train_ddp_pytorch.py --config-file CONFIG``

  The test script :download:`Download test_ddp_pytorch.py<../../scripts/action-recognition/test_ddp_pytorch.py>` can be used for
  evaluating the trained models on various datasets.

  ``python test_ddp_pytorch.py --config-file CONFIG``



Kinetics400 Dataset
-------------------

The following table lists pre-trained models trained on Kinetics400.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

  All models are trained using input size 224x224, except ``R2+1D`` models are trained and evaluated with input size of 112x112.

  ``Clip Length`` is the number of frames within an input clip. ``32 (64/2)`` means we use 32 frames, but actually the frames are formed by randomly selecting 64 consecutive frames from the video and then skipping every other frame. This strategy is widely adopted to reduce computation and memory cost.

  ``Segments`` is the number of segments used during training. For testing (reporting these numbers), we use 250 views for 2D networks (25 frames and 10-crop) and 30 views for 3D networks (10 clips and 3-crop) following the convention.


.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Kinetics400_torch.csv
   :header-rows: 1
   :class: tight-table
   :widths: 30 12 10 10 8 10 12 8


Something-Something-V2 Dataset
------------------------------

The following table lists pre-trained models trained on Something-Something-V2.

.. note::

  Our pre-trained models reproduce results from recent state-of-the-art approaches. Please check the reference paper for further information.

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Something-Something-V2_torch.csv
   :header-rows: 1
   :class: tight-table
   :widths: 30 12 10 10 8 10 12 8


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
       "Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks." \
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
.. [9] Yang, Ceyuan and Xu, Yinghao and Shi, Jianping and Dai, Bo and Zhou, Bolei. \
       "Temporal Pyramid Network for Action Recognition." \
       In Computer Vision and Pattern Recognition (CVPR), 2020.
