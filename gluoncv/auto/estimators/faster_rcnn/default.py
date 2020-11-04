"""Faster RCNN default config"""
# pylint: disable=unused-variable,missing-function-docstring,bad-whitespace,missing-class-docstring
from typing import Union, Tuple, Any
from autocfg import dataclass, field

@dataclass
class FasterRCNN:
    # Backbone network.
    backbone : str = 'resnet50_v1b'  # base feature network
    # Final R-CNN non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_thresh : float = 0.5
    # Apply R-CNN NMS to top k detection results, use -1 to disable so that every Detection
    # result is used in NMS.
    nms_topk : int = -1
    # ROI pooling mode. Currently support 'pool' and 'align'.
    roi_mode : str = 'align'
    # (height, width) of the ROI region.
    roi_size : Union[Tuple, list] = (7, 7)
    # Feature map stride with respect to original image.
    # This is usually the ratio between original image size and feature map size.
    # For FPN, use a tuple of ints.
    strides : Union[Tuple, list] = (4, 8, 16, 32, 64)
    # Clip bounding box prediction to to prevent exponentiation from overflowing.
    clip : float = 4.14

    # Anchors generation
    # ------------------
    # The width(and height) of reference anchor box.
    anchor_base_size : int = 16
    # The areas of anchor boxes.
    # We use the following form to compute the shapes of anchors:
    # .. math::
    # width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
    # height_{anchor} = size_{base} \times scale \times \sqrt{ratio}
    anchor_aspect_ratio : Union[Tuple, list] = (0.5, 1, 2)
    # The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    anchor_scales : Union[Tuple, list] = (2, 4, 8, 16, 32)

    # Allocate size for the anchor boxes as (H, W).
    # Usually we generate enough anchors for large feature map, e.g. 128x128.
    # Later in inference we can have variable input sizes,
    # at which time we can crop corresponding anchors from this large
    # anchor map so we can skip re-generating anchors for each input.
    anchor_alloc_size : Union[Tuple, list] = (384, 384)

    # number of channels used in RPN convolutional layers.
    rpn_channel : int = 256
    # IOU threshold for NMS. It is used to remove overlapping proposals.
    rpn_nms_thresh : float = 0.7
    # Maximum ground-truth number for each example. This is only an upper bound, not
    # necessarily very precise. However, using a very big number may impact the training speed.
    max_num_gt : int = 100
    # Gluon normalization layer to use. Default is none which will use frozen
    # batch normalization layer.
    norm_layer : Any = None

    # FPN Options
    # -----------
    # Whether to use FPN.
    use_fpn : bool = True
    # Number of filters for FPN output layers.
    num_fpn_filters : int = 256

    # Number of convolution layers to use in box head if batch normalization is not frozen.
    num_box_head_conv : int = 4
    # Number of filters for convolution layers in box head.
    # Only applicable if batch normalization is not frozen.
    num_box_head_conv_filters : int = 256
    # Number of hidden units for the last fully connected layer in box head.
    num_box_head_dense_filters : int = 1024

    # Input image short side size.
    image_short : int = 800
    # Maximum size of input image long side.
    image_max_size : int = 1333

    # Whether to enable custom model.
    # custom_model = True
    # Whether to use automatic mixed precision
    amp : bool = False
    # Whether to allocate memory statically.
    static_alloc : bool = False
    # whether apply transfer learning from pre-trained models, if True, override other net structures
    transfer : Union[str, None] = 'faster_rcnn_fpn_resnet50_v1b_coco'


@dataclass
class TrainCfg:
    # Whether load the imagenet pre-trained base
    pretrained_base : bool = True
    # Batch size during training
    batch_size : int = 1
    # starting epoch
    start_epoch : int = 0
    # total epoch for training
    epochs : int = 10

    # Solver
    # ------
    # Learning rate.
    lr : float = 0.01
    # Decay rate of learning rate.
    lr_decay : float = 0.1
    # Epochs at which learning rate decays
    lr_decay_epoch : Union[Tuple, list] = (20, 24)
    # Learning rate scheduler mode. options are step, poly and cosine
    lr_mode : str = 'step'
    # Number of iterations for warmup.
    lr_warmup : int = 500
    # Starging lr warmup factor.
    lr_warmup_factor : float = 1. / 3.
    # Momentum
    momentum : float = 0.9
    # Weight decay
    wd : float = 1e-4

    # RPN options
    # -----------
    # Filter top proposals before NMS in training of RPN.
    rpn_train_pre_nms : int = 12000
    # Return top proposal results after NMS in training of RPN.
    # Will be set to rpn_train_pre_nms if it is larger than rpn_train_pre_nms.
    rpn_train_post_nms : int = 2000
    # RPN box regression transition point from L1 to L2 loss.
    # Set to 0.0 to make the loss simply L1.
    rpn_smoothl1_rho : float = 0.001
    # Proposals whose size is smaller than ``min_size`` will be discarded.
    rpn_min_size : int = 1

    # R-CNN options
    # -------------
    # Number of samples for RPN targets.
    rcnn_num_samples : int = 512
    # Anchor with IOU larger than ``rcnn_pos_iou_thresh`` is regarded as positive samples.
    rcnn_pos_iou_thresh : float = 0.5
    # ``rcnn_pos_iou_thresh`` defines how many positive samples
    # (``rcnn_pos_iou_thresh * num_sample``) is to be sampled.
    rcnn_pos_ratio : float = 0.25
    # R-CNN box regression transition point from L1 to L2 loss.
    # Set to 0.0 to make the loss simply L1.
    rcnn_smoothl1_rho : float = 0.001

    # Misc
    # ----
    # log interval in terms of iterations
    log_interval : int = 100
    # Random seed to be fixed.
    seed : int = 233
    # Whether to enable verbose logging
    verbose : bool = False
    # Whether to enable mixup training
    mixup : bool = False
    # If mixup is enable, disable mixup after ```no_mixup_epochs```.
    no_mixup_epochs : int = 20
    # Number of threads for executor for scheduling ops.
    # More threads may incur higher GPU memory footprint,
    # but may speed up throughput. Note that when horovod is used,
    # it is set to 1.
    executor_threads : int = 4


@dataclass
class ValidCfg:
    # Batch size during training
    batch_size : int = 1
    # Filter top proposals before NMS in testing of RPN.
    rpn_test_pre_nms : int = 6000
    # Return top proposal results after NMS in testing of RPN.
    # Will be set to rpn_test_pre_nms if it is larger than rpn_test_pre_nms.
    rpn_test_post_nms : int = 1000
    # Epoch interval for validation
    val_interval : int = 1
    # metric, 'voc', 'voc07'
    metric : str = 'voc07'
    # iou_thresh for VOC type metrics
    iou_thresh : float = 0.5

@dataclass
class FasterRCNNCfg:
    faster_rcnn : FasterRCNN = field(default_factory=FasterRCNN)
    train : TrainCfg = field(default_factory=TrainCfg)
    valid : ValidCfg = field(default_factory=ValidCfg)
    # Dataset name. eg. 'coco', 'voc', 'voc_tiny'
    dataset : str = 'voc_tiny'
    # Path of the directory where the dataset is located.
    dataset_root : str = '~/.mxnet/datasets/'
    # Training with GPUs, you can specify (1,3) for example.
    gpus : Union[Tuple, list] = (0, 1, 2, 3)
    # Resume from previously saved parameters if not None.
    # For example, you can resume from ./faster_rcnn_xxx_0123.params.
    resume : str = ''
    # Saving parameter prefix
    save_prefix : str = ''
    # Saving parameters epoch interval, best model will always be saved.
    save_interval : int = 1
    # Use MXNet Horovod for distributed training. Must be run with OpenMPI.
    horovod : bool = False
    # Number of data workers, you can use larger number to accelerate data loading,
    # if your CPU and GPUs are powerful.
    num_workers : int = 16
    # KV store options. local, device, nccl, dist_sync, dist_device_sync,
    # dist_async are available.
    kv_store : str = 'nccl'
    # Whether to disable hybridize the model. Memory usage and speed will decrese.
    disable_hybridization : bool = False
