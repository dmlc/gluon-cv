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


Kinetics700 Dataset
-------------------

The following table lists our trained models on Kinetics700.

.. csv-table::
   :file: ./csv_tables/Action_Recognitions/Kinetics700.csv
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
