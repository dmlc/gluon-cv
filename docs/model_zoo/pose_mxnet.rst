Visualization of Inference Throughputs vs. Validation AP of COCO pre-trained models is illustrated in the following graph. Throughputs are measured with single V100 GPU and batch size 64.

.. image:: /_static/plot_help.png
  :width: 100%

.. raw:: html
   :file: ../_static/pose_throughputs.html

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

.. csv-table::
   :file: ./csv_tables/Poses/MSCOCO_Simple-Pose.csv
   :header-rows: 1
   :class: tight-table
   :widths: 33 15 15 10 15 12

Mobile Pose Models
------

By replacing the backbone network, and use pixel shuffle layer instead of deconvolution, we can have models that are very fast.

These models are suitable for edge device applications, tutorials on deployment will come soon.

Models are trained with input size 256x192, unless specified.

.. csv-table::
   :file: ./csv_tables/Poses/MSCOCO_Mobile-Pose.csv
   :header-rows: 1
   :class: tight-table
   :widths: 33 15 15 10 15 12

AlphaPose
---------
Checkout the demo tutorial here: :ref:`sphx_glr_build_examples_pose_demo_alpha_pose.py`

Alpha Pose models are evaluated with input size (320*256), unless otherwise specified. Usage is similar to simple pose section.

.. csv-table::
   :file: ./csv_tables/Poses/MSCOCO_Alpha-Pose.csv
   :header-rows: 1
   :class: tight-table
   :widths: 33 15 15 10 15 12
