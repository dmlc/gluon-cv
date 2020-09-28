"""Base Estimator"""
import os
import copy
import pickle
import logging
import warnings
from datetime import datetime
from sacred.commands import _format_config, save_config, print_config
from sacred.settings import SETTINGS
from ...utils import random as _random

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


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
    """A special hook to register the default values for decorated Estimator.

    Parameters
    ----------
    ex : sacred.Experiment
        sacred experiment object.
    """
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
    MARKER = object()
    def __init__(self, value=None):
        super(ConfigDict, self).__init__()
        self.__dict__['_freeze'] = False
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict, given {}'.format(type(value)))
        self.freeze()

    def freeze(self):
        self.__dict__['_freeze'] = True

    def is_frozen(self):
        return self.__dict__['_freeze']

    def unfreeze(self):
        self.__dict__['_freeze'] = False

    def __setitem__(self, key, value):
        if self.__dict__.get('_freeze', False):
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

    def __setstate__(self, state):
        vars(self).update(state)

    def __getstate__(self):
        return vars(self)

    __setattr__, __getattr__ = __setitem__, __getitem__


class BaseEstimator:
    """This is the base estimator for gluoncv.auto.Estimators.

    Parameters
    ----------
    config : dict
        Config in nested dict.
    logger : logging.Logger
        Optional logger for this estimator, can be `None` when default setting is used.
    reporter : callable
        The reporter for metric checkpointing.
    name : str
        Optional name for the estimator.

    Attributes
    ----------
    _logger : logging.Logger
        The customized/default logger for this estimator.
    _logdir : str
        The temporary dir for logs.
    _cfg : ConfigDict
        The configurations.

    """
    def __init__(self, config, logger=None, reporter=None, name=None):
        self._init_args = [config, logger, reporter]
        self._reporter = reporter
        name = name if isinstance(name, str) else self.__class__.__name__
        self._logger = logger if logger is not None else logging.getLogger(name)
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
                    self._logger.info('Auto resume detected previous run: %s', str(prefix))
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
        self._log_file = os.path.join(self._logdir, 'estimator.log')
        fh = logging.FileHandler(self._log_file)
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

    def state_dict(self):
        state = {
            'init_args': self._init_args,
            '__class__': self.__class__,
            'params': self.get_parameters(),
        }
        return state

    def save(self, filename):
        state = self.state_dict()
        with open(filename, 'wb') as fid:
            pickle.dump(state, fid)
        self._logger.info('Pickled to %s', filename)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fid:
            state = pickle.load(fid)
            _cls = state['__class__']
            obj = _cls(*state['init_args'])
            obj.put_parameters(state['params'])
            obj._logger.info('Unpickled from %s', filename)
            return obj
