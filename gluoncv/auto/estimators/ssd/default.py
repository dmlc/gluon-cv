from sacred import Experiment, Ingredient

from ..common import logging

ssd = Ingredient('ssd')
train = Ingredient('train')
validation = Ingredient('validation')


@ssd.config
def ssd_default():
    # Backbone network.
    backbone = 'vgg16_atrous'  # base feature network

    # Use synchronize BN across devices.
    syncbn = False

    # Input data shape, use 300, 512.
    data_shape = 300

    # Whether to use automatic mixed precision
    amp = False


@train.config
def train_cfg():
    # Batch size during training
    batch_size = 32
    # starting epoch
    start_epoch = 0
    # total epoch for training
    epochs = 240

    # Solver
    # ------
    # Learning rate.
    lr = 0.001
    # Decay rate of learning rate.
    lr_decay = 0.1
    # Epochs at which learning rate decays
    lr_decay_epoch = (160, 200)
    # Learning rate scheduler mode. options are step, poly and cosine
    lr_mode = 'step'
    # Momentum
    momentum = 0.9
    # Weight decay
    wd = 5e-4

    # Misc
    # ----
    # log interval in terms of iterations
    log_interval = 100
    seed = 233

    # Use DALI for data loading and data preprocessing in training.
    # Currently supports only COCO.
    dali = False


@validation.config
def valid_cfg():
    # Epoch interval for validation
    val_interval = 1


ex = Experiment('ssd_default', ingredients=[logging, train, validation, ssd])


@ex.config
def default_configs():
    # Dataset name. eg. 'coco', 'voc'
    dataset = 'voc'
    # Path of the directory where the dataset is located.
    dataset_root = '~/.mxnet/datasets/'
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
    num_workers = 4
