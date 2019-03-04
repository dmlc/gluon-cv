"""GluonCV Model Zoo"""
# pylint: disable=wildcard-import
from .model_zoo import get_model, get_model_list
from .model_store import pretrained_model_list
from .faster_rcnn import *
from .mask_rcnn import *
from .ssd import *
from .yolo import *
from .cifarresnet import *
from .cifarwideresnet import *
from .fcn import *
from .pspnet import *
from .deeplabv3 import *
from .deeplabv3_plus import *
from . import segbase
from .resnetv1b import *
from .se_resnet import *
from .nasnet import *
from .simple_pose.simple_pose_resnet import *

from .alexnet import *
from .densenet import *
from .inception import *
from .xception import *
from .resnet import *
from .squeezenet import *
from .vgg import *
from .mobilenet import *
from .residual_attentionnet import *
