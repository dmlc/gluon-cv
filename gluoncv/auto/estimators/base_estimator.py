"""Base Estimator"""
import os
import copy
import pickle
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sacred.commands import _format_config, save_config, print_config
from sacred.settings import SETTINGS
from ...utils import random as _random
from ...utils.filesystem import temporary_filename

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
        self._name = name
        self._logger = logger if logger is not None else logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        # reserved attributes
        self.net = None
        self.num_class = None
        self.classes = []
        self.ctx = [None]
        self.dataset = 'auto'
        self.current_epoch = 0

        # finalize the config
        r = self._ex.run('_get_config', config_updates=config, options={'--loglevel': 50, '--force': True})
        print_config(r)

        # logdir
        logdir = r.config.get('logging.logdir', None)
        self._logdir = os.path.abspath(logdir) if logdir else os.getcwd()

        prefix = name.lower() + datetime.now().strftime("-%m-%d-%Y")
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

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None, resume=False):
        """Fit with train/validation data.

        Parameters
        ----------
        train_data : pd.DataFrame or iterator
            Training data.
        val_data : pd.DataFrame or iterator, optional
            Validation data, optional. If `train_data` is DataFrame, `val_data` will be split from
            `train_data` given `train_size`.
        train_size : float
            The portion of train data split from original `train_data` if `val_data` is not provided.
        random_state : int
            Random state for splitting, for `np.random.seed`.
        resume : bool
            Whether resume from previous `fit`(if possible) or start as fresh.

        Returns
        -------
        None

        """
        if not resume:
            self.classes = train_data.classes
            self.num_class = len(self.classes)
            self._init_network()
        if not isinstance(train_data, pd.DataFrame):
            assert val_data is not None, \
                "Please provide `val_data` as we do not know how to split `train_data` of type: \
                {}".format(type(train_data))
            return self._fit(train_data, val_data) if not resume else self._resume_fit(train_data, val_data)

        if val_data is None:
            assert 0 <= train_size <= 1.0
            if random_state:
                np.random.seed(random_state)
            split_mask = np.random.rand(len(train_data)) < train_size
            train = train_data[split_mask]
            val = train_data[~split_mask]
            self._logger.info('Randomly split train_data into train[%d]/validation[%d] splits.',
                              len(train), len(val))
            return self._fit(train, val) if not resume else self._resume_fit(train, val)

        return self._fit(train_data, val_data) if not resume else self._resume_fit(train_data, val_data)

    def evaluate(self, val_data):
        """Evaluate estimator on validation data.

        Parameters
        ----------
        val_data : pd.DataFrame or iterator
            The validation data.

        """
        return self._evaluate(val_data)

    def predict(self, x):
        """Predict using this estimator.

        Parameters
        ----------
        x : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.

        """
        return self._predict(x)

    def _predict(self, x):
        raise NotImplementedError

    def _fit(self, train_data, val_data):
        raise NotImplementedError

    def _resume_fit(self, train_data, val_data):
        raise NotImplementedError

    def _evaluate(self, val_data):
        raise NotImplementedError

    def _init_network(self):
        raise NotImplementedError

    def _init_trainer(self):
        raise NotImplementedError

    def save(self, filename):
        """Save the state of this estimator to disk.

        Parameters
        ----------
        filename : str
            The file name for storing the full state.
        """
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)
        self._logger.info('Pickled to %s', filename)

    @classmethod
    def load(cls, filename):
        """Load the state from disk copy.

        Parameters
        ----------
        filename : str
            The file name to load from.
        """
        with open(filename, 'rb') as fid:
            obj = pickle.load(fid)
            obj._logger.info('Unpickled from %s', filename)
            return obj

    def __getstate__(self):
        d = self.__dict__.copy()
        try:
            import mxnet as mx
            d.pop('async_net', None)
            net = d.get('net', None)
            if isinstance(net, mx.gluon.HybridBlock):
                with temporary_filename() as tfile:
                    net.save_parameters(tfile)
                    with open(tfile, 'rb') as fi:
                        d['net'] = fi.read()
            trainer = d.get('trainer', None)
            if isinstance(trainer, mx.gluon.Trainer):
                with temporary_filename() as tfile:
                    trainer.save_states(tfile)
                    with open(tfile, 'rb') as fi:
                        d['trainer'] = fi.read()
        except ImportError:
            pass
        d['_logger'] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # logger
        self._logger = logging.getLogger(state.get('_name', self.__class__.__name__))
        self._logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self._log_file)
        self._logger.addHandler(fh)
        try:
            import mxnet as _
            net_params = state['net']
            self._init_network()
            with temporary_filename() as tfile:
                with open(tfile, 'wb') as fo:
                    fo.write(net_params)
                self.net.load_parameters(tfile)
            trainer_state = state['trainer']
            self._init_trainer()
            with temporary_filename() as tfile:
                with open(tfile, 'wb') as fo:
                    fo.write(trainer_state)
                self.trainer.load_states(tfile)
        except ImportError:
            pass
