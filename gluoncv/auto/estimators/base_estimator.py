"""Base Estimator"""
import os
import pickle
import json
import copy
import logging
import warnings
from datetime import datetime
from sacred.commands import _format_config, save_config, print_config
from sacred.settings import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from ...utils import random as _random

from ...utils.savers import save_pkl, save_json
from ...utils.loaders import load_pkl

def _get_config():
    pass

def _compare_config(r1, r2):
    r1 = copy.deepcopy(r1)
    r2 = copy.deepcopy(r2)
    ignored_keys = ('seed', 'logdir')
    for key in ignored_keys:
        r1.pop(key, None)
        r2.pop(key, None)
    return r1 == r2

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
        cls._default_config = r.config
        return cls
    return _apply


class ConfigDict(dict):
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
        self.freeze()

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
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super(ConfigDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, ConfigDict.MARKER)
        if found is ConfigDict.MARKER:
            if self.__dict__['_freeze']:
                raise KeyError(key)
            found = ConfigDict()
            super(ConfigDict, self).__setitem__(key, found)
        if isinstance(found, ConfigDict):
            found.__dict__['_freeze'] = self.__dict__['_freeze']
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__

    def __setstate__(self, state):
        vars(self).update(state)

    def __getstate__(self):
        return vars(self)


class BaseEstimator:
    def __init__(self, config, logger=None, reporter=None, name=None):
        self._reporter = reporter
        name = name if isinstance(name, str) else self.__class__.__name__
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        # finalize the config
        r = self._ex.run('_get_config', config_updates=config, options={'--loglevel': 50, '--force': True})
        print_config(r)

        # logdir
        logdir = r.config.get('logging.logdir', None)
        self._logdir = os.path.abspath(logdir) if logdir else os.getcwd()

        # try to auto resume
        prefix = None
        if r.config.get('train', {}).get('auto_resume', False):
            exists = [d for d in os.listdir(self._logdir) if d.startswith(name)]
            # latest timestamp
            exists = sorted(exists)
            prefix = exists[-1] if exists else None
            # compare config, if altered, then skip auto resume
            if prefix:
                self._ex.add_config(os.path.join(self._logdir, prefix, 'config.yaml'))
                r2 = self._ex.run('_get_config', options={'--loglevel': 50, '--force': True})
                if _compare_config(r2.config, r.config):
                    self._logger.info('Auto resume detected previous run: {}'.format(prefix))
                    r.config['seed'] = r2.config['seed']
                else:
                    prefix = None
        if not prefix:
            prefix = name + datetime.now().strftime("-%m-%d-%Y-%H-%M-%S")
        self._logdir = os.path.join(self._logdir, prefix)
        r.config['logdir'] = self._logdir
        os.makedirs(self._logdir, exist_ok=True)
        config_file = os.path.join(self._logdir, 'config.yaml')
        # log file
        log_file = os.path.join(self._logdir, 'estimator.log')
        fh = logging.FileHandler(log_file)
        self._logger.addHandler(fh)
        save_config(r.config, self._logger, config_file)

        # dot access for config
        self._cfg = ConfigDict(r.config)
        self._cfg.freeze()
        _random.seed(self._cfg.seed)

    def fit(self):
        self._fit()

    def evaluate(self):
        return self._evaluate()

    def _fit(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError

    def save(self, path):
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # with open(path, 'wb') as f:
        #     pickle.dump(self, f, protocol=4)
        save_json.save(path=path, obj=self)
        self._logger.info("estimator saved at \"%s\"" % path)

    def load(self, path):
        # with open(path, 'rb') as f:
        #     estimator = pickle.load(f)
        with open(path, 'r') as f:
            estimator = json.load(f)
        self._logger.info("estimator loaded from \"%s\"" % path)
        return estimator
