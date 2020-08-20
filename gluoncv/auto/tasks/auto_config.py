"""Utils for auto configs"""
import copy
import collections.abc

import autogluon as ag

from ..estimators.base_estimator import BaseEstimator

def cfg_to_space(cfg, space):
    for k, v in cfg.items():
        if isinstance(v, dict):
            if k not in space.keys():
                space[k] = ag.space.Dict()
            cfg_to_space(v, space[k])
        else:
            space[k] = v

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def auto_args(estimators, config):
    """Merge updated config to default, and convert to search space"""
    if not isinstance(estimators, (tuple, list)):
        estimators = [estimators]
    _cfg = {}
    for estimator in estimators:
        assert issubclass(estimator, BaseEstimator), estimator
        cfg = copy.deepcopy(estimator._default_config)
        recursive_update(_cfg, cfg)
    # user custom search space
    recursive_update(_cfg, config)
    ag_space = ag.space.Dict()
    cfg_to_space(_cfg, ag_space)
    return ag_space
