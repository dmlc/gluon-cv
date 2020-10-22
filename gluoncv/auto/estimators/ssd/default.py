"""SSD default config"""
# pylint: disable=unused-variable,missing-function-docstring
from sacred import Experiment, Ingredient

from ..common import logging

ssd = Ingredient('ssd')
train = Ingredient('train')
validation = Ingredient('validation')


@ssd.config
def ssd_default():
    # Backbone network.
    backbone = 'vgg16_atrous'  # base feature network
    # Input data shape, use 300, 512.
    data_shape = 300
    # List of convolution layer channels which is going to be appended to the base
    # network feature extractor. If `name` is `None`, this is ignored.
    filters = None
    # Sizes of anchor boxes, this should be a list of floats, in incremental order.
    # The length of `sizes` must be len(layers) + 1.
    sizes = [30, 60, 111, 162, 213, 264, 315]
    # Aspect ratios of anchors in each output layer. Its length must be equals
    # to the number of SSD output layers.
    ratios = [[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2
    # Step size of anchor boxes in each output layer.
    steps = [8, 16, 32, 64, 100, 300]
    # Use synchronize BN across devices.
    syncbn = False
    # Whether to use automatic mixed precision
    amp = False
    # Whether to enable custom model.
    # custom_model = True
    # whether apply transfer learning from pre-trained models, if True, override other net structures
    transfer = None


@train.config
def train_cfg():
    # Batch size during training
    batch_size = 32
    # starting epoch
    start_epoch = 0
    # total epoch for training
    epochs = 240
    # Learning rate.
    lr = 0.001
    # Decay rate of learning rate.
    lr_decay = 0.1
    # Epochs at which learning rate decays
    lr_decay_epoch = (160, 200)
    # Momentum
    momentum = 0.9
    # Weight decay
    wd = 5e-4
    # log interval in terms of iterations
    log_interval = 100
    # Random seed to be fixed.
    seed = 233
    # Use DALI for data loading and data preprocessing in training.
    # Currently supports only COCO.
    dali = False


@validation.config
def valid_cfg():
    # Epoch interval for validation
    val_interval = 1
    # metric, 'voc', 'voc07'
    metric = 'voc07'
    # iou_thresh for VOC type metrics
    iou_thresh = 0.5


ex = Experiment('ssd_default', ingredients=[logging, train, validation, ssd])


@ex.config
def default_configs():
    # Dataset name. eg. 'coco', 'voc', 'voc_tiny'
    dataset = 'voc_tiny'
    # Path of the directory where the dataset is located.
    dataset_root = '~/.mxnet/datasets/'
    # Training with GPUs, you can specify (1,3) for example.
    gpus = (0, 1, 2, 3)
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
    num_workers = 4
