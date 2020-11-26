"""SSD default config"""
# pylint: disable=unused-variable,missing-function-docstring,bad-whitespace,missing-class-docstring
from typing import Union, Tuple
from autocfg import dataclass, field


@dataclass
class SSD:
    # Base network name which serves as feature extraction base.
    base_network : str = 'vgg16_atrous'  # base feature network
    # Input data shape, use 300, 512.
    data_shape : int = 300
    # List of convolution layer channels which is going to be appended to the base
    # network feature extractor. If `name` is `None`, this is ignored.
    filters : Union[int, None] = None
    # Sizes of anchor boxes, this should be a list of floats, in incremental order.
    # The length of `sizes` must be len(layers) + 1.
    sizes : Union[Tuple, list] = (30, 60, 111, 162, 213, 264, 315)
    # Aspect ratios of anchors in each output layer. Its length must be equals
    # to the number of SSD output layers.
    ratios : Union[Tuple, list] = tuple([[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2)
    # Step size of anchor boxes in each output layer.
    steps : Union[Tuple, list] = (8, 16, 32, 64, 100, 300)
    # Use synchronize BN across devices.
    syncbn : bool = False
    # Whether to use automatic mixed precision
    amp : bool = False
    # Whether to enable custom model.
    # custom_model = True
    # whether apply transfer learning from pre-trained models, if True, override other net structures
    transfer : Union[str, None] = 'ssd_512_resnet50_v1_coco'


@dataclass
class TrainCfg:
    # Batch size during training
    batch_size : int = 16
    # starting epoch
    start_epoch : int = 0
    # total epoch for training
    epochs : int = 20
    # Learning rate.
    lr : float = 0.001
    # Decay rate of learning rate.
    lr_decay : float = 0.1
    # Epochs at which learning rate decays
    lr_decay_epoch : Union[Tuple, list] = (160, 200)
    # Momentum
    momentum : float = 0.9
    # Weight decay
    wd : float = 5e-4
    # log interval in terms of iterations
    log_interval : int = 100
    # Random seed to be fixed.
    seed : int = 233
    # Use DALI for data loading and data preprocessing in training.
    # Currently supports only COCO.
    dali : bool = False


@dataclass
class ValidCfg:
    # Batch size during training
    batch_size : int = 16
    # Epoch interval for validation
    val_interval : int = 1
    # metric, 'voc', 'voc07'
    metric : str = 'voc07'
    # iou_thresh for VOC type metrics
    iou_thresh : float = 0.5

@dataclass
class SSDCfg:
    ssd : SSD = field(default_factory=SSD)
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
    # Saving parameters epoch interval, best model will always be saved.
    save_interval : int = 1
    # Use MXNet Horovod for distributed training. Must be run with OpenMPI.
    horovod : bool = False
    # Number of data workers, you can use larger number to accelerate data loading,
    # if your CPU and GPUs are powerful.
    num_workers : int = 4
