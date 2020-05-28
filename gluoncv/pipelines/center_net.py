"""CenterNet Estimator"""
from .base import BaseEstimator, set_default
from yacs.config import CfgNode as CN

def get_cfg_defaults():
    _C = CN()

    _C.SYSTEM = CN()
    # Number of GPUS to use in the experiment
    _C.SYSTEM.NUM_GPUS = 8
    # Number of workers for doing things
    _C.SYSTEM.NUM_WORKERS = 4

    _C.TRAIN = CN()
    # A very important hyperparameter
    _C.TRAIN.HYPERPARAMETER_1 = 0.1
    # The all important scales for the stuff
    _C.TRAIN.SCALES = (2, 4, 8, 16)
    return _C.clone()

@set_default(get_cfg_defaults)
class CenterNetEstimator(BaseEstimator):
    """CenterNet Estimator."""
    def __init__(self, config=None, reporter=None, logdir=None):
        super(CenterNetEstimator, self).__init__(config, reporter, logdir)

    def fit(self, train_data):
        self.finalize_config()
