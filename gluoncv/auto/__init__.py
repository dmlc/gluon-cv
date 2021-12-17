"""GluonCV auto"""
import logging
from .estimators import *

logger = logging.getLogger(__name__)
logger.warning("We plan to deprecate auto from gluoncv on release 0.12.0. Please consider using autogluon.vision instead, which provides same functionality.")
