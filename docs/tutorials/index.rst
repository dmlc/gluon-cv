Tutorials
=========

Interested in getting started in a new CV area? Here are some tutorials to help get started.

Image Classification
--------------------

.. container:: cards

    .. card::
        :title: Pre-trained Models on CIFAR10
        :link: ../build/examples_classification/demo_cifar10.html

        Basics on how to use pretrained models on CIFAR10 and apply to real images

    .. card::
        :title: Training on CIFAR10
        :link: ../build/examples_classification/dive_deep_cifar10.html

        Hands on classification model training on CIFAR10

    .. card::
        :title: Pre-trained Models on ImageNet
        :link: ../build/examples_classification/demo_imagenet.html

        Basics on how to use pretrained models on ImageNet and apply to real images

    .. card::
        :title: Transfer Learning with Your Own Dataset
        :link: ../build/examples_classification/transfer_learning_minc.html

        Train on your own dataset with ImageNet pre-trained models.

    .. card::
        :title: Training on ImageNet
        :link: ../build/examples_classification/dive_deep_imagenet.html

        Hands on classification model training on ImageNet


Object Detection
----------------

.. container:: cards

    .. card::
        :title: Pre-trained SSD Models
        :link: ../build/examples_detection/demo_ssd.html

        Detect objects in real-world images with pre-trained SSD models

    .. card::
        :title: Training SSD on Pascal VOC Dataset
        :link: ../build/examples_detection/train_ssd_voc.html

        Hands on SSD model training on Pascal VOC Dataset

    .. card::
        :title: Tips for SSD Model Training
        :link: ../build/examples_detection/train_ssd_advanced.html

        Training tips to boost your SSD Model performance.

    .. card::
        :title: Pre-trained Faster R-CNN Models
        :link: ../build/examples_detection/demo_faster_rcnn.html

        Detect objects in real-world images with pre-trained Faster R-CNN models

    .. card::
        :title: Training Faster R-CNN on Pascal VOC
        :link: ../build/examples_detection/train_faster_rcnn_voc.html

        End-to-end Faster R-CNN Training on Pascal VOC

    .. card::
        :title: Pre-trained YOLO Models
        :link: ../build/examples_detection/demo_yolo.html

        Detect objects in real-world images with pre-trained YOLO models

    .. card::
        :title: Training YOLOv3 on Pascal VOC
        :link: ../build/examples_detection/train_yolo_v3.html

        Hands on YOLOv3 model training on Pascal VOC Dataset

    .. card::
        :title: Finetune a Pre-trained Model
        :link: ../build/examples_detection/finetune_detection.html

        Finetune a pre-trained model on your own dataset.

    .. card::
        :title: Object Detection from Webcam
        :link: ../build/examples_detection/demo_webcam.html

        Run an object detection model from your webcam.

    .. card::
        :title: Skip Finetuning by reusing part of pre-trained model
        :link: ../build/examples_detection/skip_fintune.html


Instance Segmentation
---------------------

.. container:: cards

    .. card::
        :title: Pre-trained Mask R-CNN Models
        :link: ../build/examples_instance/demo_mask_rcnn.html

        Perform instance segmentation on real-world images with pre-trained Mask R-CNN models

    .. card::
        :title: Training Mask R-CNN on MS COCO
        :link: ../build/examples_instance/train_mask_rcnn_coco.html

        Hands on Mask R-CNN model training on MS COCO dataset


Semantic Segmentation
---------------------

.. container:: cards

    .. card::
        :title: Pre-trained FCN Models
        :link: ../build/examples_segmentation/demo_fcn.html

        Perform semantic segmentation on real-world images with pre-trained FCN models

    .. card::
        :title: Training FCN on Pascal VOC
        :link: ../build/examples_segmentation/train_fcn.html

        Hands on FCN model training on Pascal VOC dataset

    .. card::
        :title: Pre-trained PSPNet Models
        :link: ../build/examples_segmentation/demo_psp.html

        Perform semantic segmentation in real-world images with pre-trained PSPNet models

    .. card::
        :title: Training PSPNet on ADE20K
        :link: ../build/examples_segmentation/train_psp.html

        Hands on Mask R-CNN model training on ADE20K dataset

    .. card::
        :title: Pre-trained DeepLabV3 Models
        :link: ../build/examples_segmentation/demo_deeplab.html

        Perform instance segmentation in real-world images with pre-trained DeepLabV3 models

    .. card::
        :title: Getting SOTA Results on Pascal VOC
        :link: ../build/examples_segmentation/voc_sota.html

        Hands on DeepLabV3 model training on Pascal VOC dataset, and achieves
        state-of-the-art accuracy.

Pose Estimation
---------------------

.. container:: cards

    .. card::
        :title: Pre-trained Simple Pose Models
        :link: ../build/examples_pose/demo_simple_pose.html

        Estimate human pose in real-world images with pre-trained Simple Pose models

Action Recognition
---------------------

.. container:: cards

    .. card::
        :title: Pre-trained TSN Models on UCF101
        :link: ../build/examples_action_recognition/demo_ucf101.html

        Recognize human actions in real-world videos with pre-trained TSN-VGG16 models

    .. card::
        :title: Training TSN models on UCF101
        :link: ../build/examples_action_recognition/dive_deep_ucf101.html

        Hands on TSN-VGG16 action recognition model training on UCF101 dataset

Dataset Preparation
-------------------

.. container:: cards

    .. card::
        :title: Prepare ADE20K Dataset
        :link: ../build/examples_datasets/ade20k.html

    .. card::
        :title: Prepare MS COCO Dataset
        :link: ../build/examples_datasets/mscoco.html

    .. card::
        :title: Prepare Cityscapes Dataset
        :link: ../build/examples_datasets/cityscapes.html

    .. card::
        :title: Prepare Pascal VOC Dataset
        :link: ../build/examples_datasets/pascal_voc.html

    .. card::
        :title: Prepare Custom Dataset for Object Detection
        :link: ../build/examples_datasets/detection_custom.html

    .. card::
        :title: Prepare ImageNet Dataset
        :link: ../build/examples_datasets/imagenet.html

    .. card::
        :title: Prepare ImageNet Dataset in ImageRecord Format
        :link: ../build/examples_datasets/recordio.html

    .. card::
        :title: Prepare UCF101 Dataset
        :link: ../build/examples_datasets/ucf101.html


Deployment
----------

.. container:: cards

    .. card::
        :title: Export Models into JSON
        :link: ../build/examples_deployment/export_network.html

    .. card::
        :title: C++ Inference with GluonCV
        :link: ../build/examples_deployment/cpp_inference.html

    .. card::
        :title: Inference with Quantized Models
        :link: ../build/examples_deployment/int8_inference.html

.. toctree::
    :hidden:
    :maxdepth: 2

    ../build/examples_classification/index
    ../build/examples_detection/index
    ../build/examples_instance/index
    ../build/examples_segmentation/index
    ../build/examples_pose/index
    ../build/examples_action_recognition/index
    ../build/examples_datasets/index
    ../build/examples_deployment/index
