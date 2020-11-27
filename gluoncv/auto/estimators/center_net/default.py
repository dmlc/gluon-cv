"""Default configs for center net"""
# pylint: disable=bad-whitespace,missing-class-docstring,bad-indentation
import os
from typing import Union, Tuple
from autocfg import dataclass, field

@dataclass
class CenterNetHead:
    bias : float = -2.19          # use bias = -log((1 - 0.1) / 0.1)
    wh_outputs : int = 2          # wh head channel
    reg_outputs : int = 2         # regression head channel
    head_conv_channel : int = 64  # additional conv channel

@dataclass
class CenterNet:
  base_network : str = 'dla34_deconv'  # base feature network
  heads : CenterNetHead = field(default_factory=CenterNetHead)
  scale : float = 4.0  # output vs input scaling ratio, e.g., input_h // feature_h
  topk : int = 100  # topk detection results will be kept after inference
  root : str = os.path.expanduser(os.path.join('~', '.mxnet', 'models'))  # model zoo root dir
  wh_weight : float = 0.1  # Loss weight for width/height
  center_reg_weight : float = 1.0  # Center regression loss weight
  data_shape : Tuple[int, int] = (512, 512)
  # use the pre-trained detector for transfer learning(use preset, ignore other network settings)
  transfer : str = 'center_net_resnet50_v1b_coco'

@dataclass
class TrainCfg:
    pretrained_base : bool = True  # whether load the imagenet pre-trained base
    batch_size : int = 16
    epochs : int = 15
    lr : float = 1.25e-4  # learning rate
    lr_decay : float = 0.1  # decay rate of learning rate.
    lr_decay_epoch : Tuple[int, int] = (90, 120)  # epochs at which learning rate decays
    lr_mode : str = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    warmup_lr : float = 0.0  # starting warmup learning rate.
    warmup_epochs : int = 0  # number of warmup epochs
    num_workers : int = 16  # cpu workers, the larger the more processes used
    start_epoch : int = 0
    momentum : float = 0.9  # SGD momentum
    wd : float = 1e-4  # weight decay
    log_interval : int = 100  # logging interval

@dataclass
class ValidCfg:
    flip_test : bool = True  # use flip in validation test
    nms_thresh : Union[float, int] = 0  # 0 means disable
    nms_topk : int = 400  # pre nms topk
    post_nms : int = 100  # post nms topk
    num_workers : int = 16  # cpu workers, the larger the more processes used
    batch_size : int = 8  # validation batch size
    interval : int = 1  # validation epoch interval, for slow validations
    metric : str = 'voc07' # metric, 'voc', 'voc07'
    iou_thresh : float = 0.5 # iou_thresh for VOC type metrics

@dataclass
class CenterNetCfg:
    center_net : CenterNet = field(default_factory=CenterNet)
    train : TrainCfg = field(default_factory=TrainCfg)
    valid : ValidCfg = field(default_factory=ValidCfg)
    gpus : Union[Tuple, list] = (0, 1, 2, 3)  # gpu individual ids, not necessarily consecutive
