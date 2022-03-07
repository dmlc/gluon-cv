"""AutoML Tasks"""
import logging

from .image_classification import *
from .object_detection import *
# from .utils import *

logger = logging.getLogger(__name__)
msg = (
    "We plan to deprecate auto from gluoncv on release 0.12.0."
    "Please consider using autogluon.vision instead, which provides the same functionality."
    "https://auto.gluon.ai/stable/tutorials/image_prediction/index.html"
)
logger.warning(msg)
