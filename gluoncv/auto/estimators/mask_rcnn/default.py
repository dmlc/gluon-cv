"""Mask RCNN default config"""
# pylint: disable=unused-variable,missing-function-docstring,bad-whitespace,missing-class-docstring
# from typing import Union, Tuple
from autocfg import dataclass, field

@dataclass
class MaskRCNN:
    # Backbone network.
    backbone = 'resnet50_v1b'  # base feature network
    # Final R-CNN non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_thresh = 0.5
    # Apply R-CNN NMS to top k detection results, use -1 to disable so that every Detection
    # result is used in NMS.
    nms_topk = -1
    # Only return top `post_nms` detection results, the rest is discarded.
    # Set to -1 to return all detections.
    post_nms = -1
    # ROI pooling mode. Currently support 'pool' and 'align'.
    roi_mode = 'align'
    # (height, width) of the ROI region.
    roi_size = (7, 7)
    # Feature map stride with respect to original image.
    # This is usually the ratio between original image size and feature map size.
    # For FPN, use a tuple of ints.
    strides = (4, 8, 16, 32, 64)
    # Clip bounding box prediction to to prevent exponentiation from overflowing.
    clip = 4.14

    # Anchors generation
    # ------------------
    # The width(and height) of reference anchor box.
    anchor_base_size = 16
    # The areas of anchor boxes.
    # We use the following form to compute the shapes of anchors:
    # .. math::
    # width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
    # height_{anchor} = size_{base} \times scale \times \sqrt{ratio}
    anchor_aspect_ratio = (0.5, 1, 2)
    # The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    anchor_scales = (2, 4, 8, 16, 32)

    # Allocate size for the anchor boxes as (H, W).
    # Usually we generate enough anchors for large feature map, e.g. 128x128.
    # Later in inference we can have variable input sizes,
    # at which time we can crop corresponding anchors from this large
    # anchor map so we can skip re-generating anchors for each input.
    anchor_alloc_size = (384, 384)

    # number of channels used in RPN convolutional layers.
    rpn_channel = 256
    # IOU threshold for NMS. It is used to remove overlapping proposals.
    rpn_nms_thresh = 0.7
    # Maximum ground-truth number for each example. This is only an upper bound, not
    # necessarily very precise. However, using a very big number may impact the training speed.
    max_num_gt = 100
    # Gluon normalization layer to use. Default is none which will use frozen
    # batch normalization layer.
    norm_layer = None

    # FPN Options
    # -----------
    # Whether to use FPN.
    use_fpn = True
    # Number of filters for FPN output layers.
    num_fpn_filters = 256

    # Number of convolution layers to use in box head if batch normalization is not frozen.
    num_box_head_conv = 4
    # Number of filters for convolution layers in box head.
    # Only applicable if batch normalization is not frozen.
    num_box_head_conv_filters = 256
    # Number of hidden units for the last fully connected layer in box head.
    num_box_head_dense_filters = 1024

    # Input image short side size.
    image_short = 800
    # Maximum size of input image long side.
    image_max_size = 1333

    # Whether to enable custom model.
    custom_model = True
    # Whether to use automatic mixed precision
    amp = False
    # Whether to allocate memory statically.
    static_alloc = False

    # Ratio of mask output roi / input roi.
    # For model with FPN, this is typically 2.
    target_roi_scale = 2
    # Number of convolution blocks before deconv layer for mask head.
    # For FPN network this is typically 4.
    num_mask_head_convs = 4


@dataclass
class TrainCfg:
    # Whether load the imagenet pre-trained base
    pretrained_base = True
    # Batch size during training
    batch_size = 1
    # starting epoch
    start_epoch = 0
    # total epoch for training
    epochs = 26

    # Solver
    # ------
    # Learning rate.
    lr = 0.01
    # Decay rate of learning rate.
    lr_decay = 0.1
    # Epochs at which learning rate decays
    lr_decay_epoch = (20, 24)
    # Learning rate scheduler mode. options are step, poly and cosine
    lr_mode = 'step'
    # Number of iterations for warmup.
    lr_warmup = 500
    # Starging lr warmup factor.
    lr_warmup_factor = 1. / 3.
    # Gradient clipping.
    clip_gradient = -1
    # Momentum
    momentum = 0.9
    # Weight decay
    wd = 1e-4

    # RPN options
    # -----------
    # Filter top proposals before NMS in training of RPN.
    rpn_train_pre_nms = 12000
    # Return top proposal results after NMS in training of RPN.
    # Will be set to rpn_train_pre_nms if it is larger than rpn_train_pre_nms.
    rpn_train_post_nms = 2000
    # RPN box regression transition point from L1 to L2 loss.
    # Set to 0.0 to make the loss simply L1.
    rpn_smoothl1_rho = 0.001
    # Proposals whose size is smaller than ``min_size`` will be discarded.
    rpn_min_size = 1

    # R-CNN options
    # -------------
    # Number of samples for RPN targets.
    rcnn_num_samples = 512
    # Anchor with IOU larger than ``rcnn_pos_iou_thresh`` is regarded as positive samples.
    rcnn_pos_iou_thresh = 0.5
    # ``rcnn_pos_iou_thresh`` defines how many positive samples
    # (``rcnn_pos_iou_thresh * num_sample``) is to be sampled.
    rcnn_pos_ratio = 0.25
    # R-CNN box regression transition point from L1 to L2 loss.
    # Set to 0.0 to make the loss simply L1.
    rcnn_smoothl1_rho = 0.001

    # Misc
    # ----
    # log interval in terms of iterations
    log_interval = 100
    seed = 233
    # Whether to enable verbose logging
    verbose = False
    # Number of threads for executor for scheduling ops.
    # More threads may incur higher GPU memory footprint,
    # but may speed up throughput. Note that when horovod is used,
    # it is set to 1.
    executor_threads = 4


@dataclass
class ValidCfg:
    # Filter top proposals before NMS in testing of RPN.
    rpn_test_pre_nms = 6000
    # Return top proposal results after NMS in testing of RPN.
    # Will be set to rpn_test_pre_nms if it is larger than rpn_test_pre_nms.
    rpn_test_post_nms = 1000
    # Epoch interval for validation
    val_interval = 1

@dataclass
class MaskRCNNCfg:
    mask_rcnn : MaskRCNN = field(default_factory=MaskRCNN)
    train : TrainCfg = field(default_factory=TrainCfg)
    valid : ValidCfg = field(default_factory=ValidCfg)    # Dataset name. eg. 'coco', 'voc'
    dataset = 'coco'
    # Training with GPUs, you can specify (1,3) for example.
    gpus = (0,)
    # Resume from previously saved parameters if not None.
    # For example, you can resume from ./faster_rcnn_xxx_0123.params.
    resume = ''
    # Saving parameter prefix
    save_prefix = ''
    # Saving parameters epoch interval, best model will always be saved.
    save_interval = 1
    # Use MXNet Horovod for distributed training. Must be run with OpenMPI.
    horovod = False
    # Number of data workers, you can use larger number to accelerate data loading,
    # if your CPU and GPUs are powerful.
    num_workers = 16
    # KV store options. local, device, nccl, dist_sync, dist_device_sync,
    # dist_async are available.
    kv_store = 'nccl'
    # Whether to disable hybridize the model. Memory usage and speed will decrese.
    disable_hybridization = False
    # Use NVIDIA MSCOCO API. Make sure you install first.
    use_ext = False
