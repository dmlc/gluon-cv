"""Base Estimator"""
import os
import logging
from datetime import datetime
from sacred.commands import _format_config, save_config, print_config
# from sacred.settings import SETTINGS
# SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from ...utils import random as _random

def _fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__

def _get_config():
    pass

def set_default(ex):
    def _apply(cls):
        # docstring
        cls.__doc__ = str(cls.__doc__) if cls.__doc__ else ''
        cls.__doc__ += ("\n\nParameters\n"
                        "----------\n"
                        "config : str, dict\n"
                        "  Config used to override default configurations. \n"
                        "  If `str`, assume config file (.yml, .yaml) is used. \n"
                        "logger : logger, default is `None`.\n"
                        "  If not `None`, will use default logging object.\n"
                        "logdir : str, default is None.\n"
                        "  Directory for saving logs. If `None`, current working directory is used.\n")
        cls.__doc__ += '\nDefault configurations: \n----------\n'
        ex.command(_get_config, unobserved=True)
        r = ex.run('_get_config', options={'--loglevel': 50})
        if 'seed' in r.config:
            r.config.pop('seed')
        cls.__doc__ += str("\n".join(_format_config(r.config, r.config_modifications).splitlines()[1:]))
        # default config
        cls._ex = ex
        return cls
    return _apply


class BaseEstimator:
    def __init__(self, config, logger=None, logdir=None):
        self._ex.add_config(config)
        r = self._ex.run('_get_config', options={'--loglevel': 50})
        self._cfg = r.config
        self._log = logger if logger is not None else logging.getLogger(__name__)
        self._logdir = os.path.abspath(logdir) if logdir else os.getcwd()

    def fit(self):
        # finalize the config
        r = self._ex.run('_get_config', options={'--loglevel': 50})
        print_config(r)
        config_fn = _fullname(self) + datetime.now().strftime("-%m-%d-%Y-%H-%M-%S.yaml")
        config_file = os.path.join(self._logdir, config_fn)
        save_config(r.config, self._log, config_file)
        self._cfg = r.config
        _random.seed(self._cfg.seed)
        self._fit()

    def _fit(self):
        raise NotImplementedError
