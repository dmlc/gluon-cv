"""YOLO default config"""
# pylint: disable=unused-variable,missing-function-docstring,bad-whitespace,missing-class-docstring
from typing import Union, Tuple
from autocfg import dataclass, field


@dataclass
class YOLOv3:
    # Base network name which serves as feature extraction base.
    base_network : str = 'darknet53'
    # List of convolution layer channels which is going to be appended to the
    # base network feature extractor. If `name` is `None`, this is ignored.
    filters : Union[Tuple, list] = (512, 256, 128)
    # The anchor setting.
    anchors : Union[Tuple, list] = tuple([[10, 13, 16, 30, 33, 23],
                                          [30, 61, 62, 45, 59, 119],
                                          [116, 90, 156, 198, 373, 326]])
    # Strides of feature map.
    strides : Union[Tuple, list] = (8, 16, 32)
    # Input data shape for evaluation, use 320, 416, 608...
    # Training is with random shapes from (320 to 608).
    data_shape : int = 416
    # Use synchronize BN across devices.
    syncbn : bool = False
    # Use fixed size(data-shape) throughout the training, which will be faster
    # and require less memory. However, final model will be slightly worse.
    no_random_shape : bool = False
    # Use MXNet AMP for mixed precision training.
    amp : bool = False
    # Whether to enable custom model.
    # custom_model = True
    # whether apply transfer learning from pre-trained models, if True, override other net structures
    transfer : Union[str, None] = 'yolo3_darknet53_coco'
    # NMS settings
    nms_thresh : Union[float, int] = 0.45
    nms_topk : int = 400


@dataclass
class TrainCfg:
    # Training mini-batch size
    batch_size : int = 16
    # Training epochs.
    epochs : int = 20
    # Starting epoch for resuming, default is 0 for new training.
    # You can specify it to 100 for example to start from 100 epoch.
    start_epoch : int = 0
    # Learning rate.
    lr : float = 0.001
    # learning rate scheduler mode. options are step, poly and cosine.
    lr_mode : str = 'step'
    # decay rate of learning rate.
    lr_decay : float = 0.1
    # interval for periodic learning rate decays.
    lr_decay_period : int = 0
    # epochs at which learning rate decays.
    lr_decay_epoch : Union[Tuple, list] = (160, 180)
    # starting warmup learning rate.
    warmup_lr : float = 0.0
    # number of warmup epochs.
    warmup_epochs : int = 0
    # SGD momentum.
    momentum : float = 0.9
    # Weight decay.
    wd : float = 0.0005
    # Logging mini-batch interval.
    log_interval : int = 100
    # Random seed to be fixed.
    seed : int = 233
    # Training images. Use -1 to automatically get the number.
    num_samples : int = -1
    # whether to remove weight decay on bias, and beta/gamma for batchnorm layers.
    no_wd : bool = False
    # whether to enable mixup.
    mixup : bool = False
    # Disable mixup training if enabled in the last N epochs.
    no_mixup_epochs : int = 20
    # Use label smoothing.
    label_smooth : bool = False
    early_stop_patience : int = -1  # epochs with no improvement after which train is early stopped, negative: disabled
    early_stop_min_delta : float = 0.001  # ignore changes less than min_delta for metrics
    # the baseline value for metric, training won't stop if not reaching baseline
    early_stop_baseline : Union[float, int] = 0.0
    early_stop_max_value : Union[float, int] = 1.0  # early stop if reaching max value instantly


@dataclass
class ValidCfg:
    # Batch size during training
    batch_size : int = 16
    # Epoch interval for validation, increase the number
    # will reduce the training time if validation is slow.
    val_interval : int = 1
    # metric, 'voc', 'voc07'
    metric : str = 'voc07'
    # iou_thresh for VOC type metrics
    iou_thresh : float = 0.5


@dataclass
class YOLOv3Cfg:
    yolo3 : YOLOv3 = field(default_factory=YOLOv3)
    train : TrainCfg = field(default_factory=TrainCfg)
    valid : ValidCfg = field(default_factory=ValidCfg)
    # Training dataset. eg. 'coco', 'voc', 'voc_tiny'
    dataset : str = 'voc_tiny'
    # Path of the directory where the dataset is located.
    dataset_root : str = '~/.mxnet/datasets/'
    # Number of data workers, you can use larger number to
    # accelerate data loading, if you CPU and GPUs are powerful.
    num_workers : int = 4
    # Training with GPUs, you can specify (1,3) for example.
    gpus : Union[Tuple, list] = (0, 1, 2, 3)
    # Resume from previously saved parameters if not None.
    # For example, you can resume from ./yolo3_xxx_0123.params
    resume : str = ''
    # Saving parameter prefix.
    save_prefix : str = ''
    # Saving parameters epoch interval, best model will always be saved.
    save_interval : int = 10
    # Use MXNet Horovod for distributed training. Must be run with OpenMPI.
    horovod : bool = False
