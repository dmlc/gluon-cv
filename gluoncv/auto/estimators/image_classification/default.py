"""Default configs for image classification"""
# pylint: disable=bad-whitespace,missing-class-docstring
from typing import Union, Tuple
from autocfg import dataclass, field


@dataclass
class ImageClassification:
    model : str = 'resnet50_v1'
    use_pretrained : bool = True
    use_gn : bool = False
    batch_norm : bool = False
    use_se : bool = False
    last_gamma : bool = False

@dataclass
class TrainCfg:
    pretrained_base : bool = True  # whether load the imagenet pre-trained base
    batch_size : int = 128
    epochs : int = 10
    lr : float = 0.1  # learning rate
    lr_decay : float = 0.1  # decay rate of learning rate.
    lr_decay_period : int = 0
    lr_decay_epoch : str = '40, 60'  # epochs at which learning rate decays
    lr_mode : str = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    warmup_lr : float = 0.0  # starting warmup learning rate.
    warmup_epochs : int = 0  # number of warmup epochs
    num_training_samples : int = 1281167
    num_workers : int = 4
    wd : float = 0.0001
    momentum : float = 0.9
    teacher : Union[None, str] = None
    hard_weight : float = 0.5
    dtype : str = 'float32'
    input_size : int = 224
    crop_ratio : float = 0.875
    use_rec : bool = False
    rec_train : str = '~/.mxnet/datasets/imagenet/rec/train.rec'
    rec_train_idx : str = '~/.mxnet/datasets/imagenet/rec/train.idx'
    rec_val : str = '~/.mxnet/datasets/imagenet/rec/val.rec'
    rec_val_idx : str = '~/.mxnet/datasets/imagenet/rec/val.idx'
    data_dir : str = '~/.mxnet/datasets/imagenet'
    mixup : bool = False
    no_wd : bool = False
    label_smoothing : bool = False
    temperature : Union[int, float] = 20
    hard_weight : float = 0.5
    resume_epoch : int = 0
    mixup_alpha : float = 0.2
    mixup_off_epoch : int = 0
    log_interval : int = 50
    mode : str = ''
    start_epoch : int = 0
    transfer_lr_mult : float = 0.01  # reduce the backbone lr_mult to avoid quickly destroying the features
    output_lr_mult : float = 0.1  # the learning rate multiplier for last fc layer if trained with transfer learning

@dataclass
class ValidCfg:
    batch_size : int = 128
    num_workers : int = 4

@dataclass
class ImageClassificationCfg:
    img_cls : ImageClassification = field(default_factory=ImageClassification)
    train : TrainCfg = field(default_factory=TrainCfg)
    valid : ValidCfg = field(default_factory=ValidCfg)
    gpus : Union[Tuple, list] = (0, )  # gpu individual ids, not necessarily consecutive
