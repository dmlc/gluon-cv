"""Base Estimator"""
import os
import logging
import warnings
from datetime import datetime
from sacred.commands import _format_config, save_config, print_config
from sacred.settings import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from ...utils import random as _random

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


class DotDict(dict):
    MARKER = object()
    """The view of a config dict where keys can be accessed like attribute, it also prevents
    naive modifications to the key-values.

    Parameters
    ----------
    config : dict
        The sacred configuration dict.

    Attributes
    ----------
    __dict__ : type
        The internal config as a `__dict__`.

    """
    def __init__(self, value=None):
        self.__dict__['_freeze'] = False
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def freeze(self):
        self.__dict__['_freeze'] = True

    def is_frozen(self):
        return self.__dict__['_freeze']

    def unfreeze(self):
        self.__dict__['_freeze'] = False

    def __setitem__(self, key, value):
        if self.__dict__['_freeze']:
            msg = ('You are trying to modify the config to "{}={}" after initialization, '
                   ' this may result in unpredictable behaviour'.format(key, value))
            warnings.warn(msg)
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        super(DotDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, DotDict.MARKER)
        if found is DotDict.MARKER:
            found = DotDict()
            super(DotDict, self).__setitem__(key, found)
        if isinstance(found, DotDict):
            found.__dict__['_freeze'] = self.__dict__['_freeze']
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


class BaseEstimator:
    def __init__(self, config, logger=None):
        self._log = logger if logger is not None else logging.getLogger(__name__)

        # finalize the config
        r = self._ex.run('_get_config', config_updates=config, options={'--loglevel': 50})
        print_config(r)
        config_fn = self.__class__.__name__ + datetime.now().strftime("-%m-%d-%Y-%H-%M-%S.yaml")

        logdir = r.config.get('logdir', None)
        self._logdir = os.path.abspath(logdir) if logdir else os.getcwd()
        config_file = os.path.join(self._logdir, config_fn)
        save_config(r.config, self._log, config_file)
        self._cfg = DotDict(r.config)
        self._cfg.freeze()
        _random.seed(self._cfg.seed)

    def fit(self):
        self._fit()

    def evaluate(self):
        self._evaluate()

    def _fit(self):
        raise NotImplementedError
