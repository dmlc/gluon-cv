"""Base Estimator"""
import os
import math
import pickle
import io
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from ...utils import random as _random
from ...utils.filesystem import temporary_filename

logging.basicConfig(level=logging.INFO)

def set_default(cfg):
    """A special hook to register the default values for decorated Estimator.

    Parameters
    ----------
    cfg : autocfg.dataclass
        Default config as dataclass object
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
                        "  If not `None`, will use default logging object.\n")
        cls.__doc__ += '\nDefault configurations: \n--------------------\n'
        sio = io.StringIO()
        cfg.save(sio)
        cls.__doc__ += '\n' + sio.getvalue()
        cls._default_cfg = cfg
        cls._default_cfg.freeze()
        return cls
    return _apply


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
    _cfg : autocfg.dataclass
        The configurations.

    """
    def __init__(self, config, logger=None, reporter=None, name=None):
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

        # logdir
        logdir = config.pop('log_dir', None) if isinstance(config, dict) else None
        if logdir:
            self._logdir = os.path.abspath(logdir)
        else:
            self._logdir = os.path.join(os.getcwd(), name.lower() + datetime.now().strftime("-%m-%d-%Y"))

        # finalize config
        cfg = self._default_cfg.merge(config)  # config can be dict or yaml file
        diffs = self._default_cfg.diff(cfg)
        if diffs:
            self._logger.info('modified configs(<old> != <new>): {')
            for diff in diffs:
                self._logger.info(diff)
            self._logger.info('}')
        self._cfg = cfg

        os.makedirs(self._logdir, exist_ok=True)
        config_file = os.path.join(self._logdir, 'config.yaml')
        # log file
        self._log_file = os.path.join(self._logdir, 'estimator.log')
        fh = logging.FileHandler(self._log_file)
        self._logger.addHandler(fh)
        # save_config(r.config, self._logger, config_file)
        self._cfg.save(config_file)
        self._logger.info('Saved config to %s', config_file)

        # freeze config
        self._cfg.freeze()
        seed = self._cfg.get('seed', np.random.randint(1000000))
        _random.seed(seed)

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None,
            resume=False, time_limit=None):

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
        time_limit : int, default is None
            The wall clock time limit(second) for fit process, if `None`, time limit is not enforced.
            If `fit` takes longer than `time_limit`, the process will terminate early and return the
            model prematurally.
            Due to callbacks and additional validation functions, the `time_limit` may not be very precise
            (few minutes allowance), but you can use it to safe-guard a very long training session.

        Returns
        -------
        None

        """
        if time_limit is None:
            time_limit = math.inf
        elif not isinstance(time_limit, (int, float)):
            raise TypeError(f'Invalid type `time_limit={time_limit}`, int/float or None expected')
        if not resume:
            self.classes = train_data.classes
            self.num_class = len(self.classes)
            self._init_network()
        if not isinstance(train_data, pd.DataFrame):
            assert val_data is not None, \
                "Please provide `val_data` as we do not know how to split `train_data` of type: \
                {}".format(type(train_data))
            return self._fit(train_data, val_data, time_limit=time_limit) if not resume \
                else self._resume_fit(train_data, val_data, time_limit=time_limit)

        os.makedirs(self._logdir, exist_ok=True)
        if val_data is None:
            assert 0 <= train_size <= 1.0
            if random_state:
                np.random.seed(random_state)
            split_mask = np.random.rand(len(train_data)) < train_size
            train = train_data[split_mask]
            val = train_data[~split_mask]
            self._logger.info('Randomly split train_data into train[%d]/validation[%d] splits.',
                              len(train), len(val))
            return self._fit(train, val, time_limit=time_limit) if not resume else \
                self._resume_fit(train, val, time_limit=time_limit)

        return self._fit(train_data, val_data, time_limit=time_limit) if not resume else \
            self._resume_fit(train_data, val_data, time_limit=time_limit)

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

    def predict_feature(self, x):
        """Predict intermediate features using this estimator.

        Parameters
        ----------
        x : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.

        """
        return self._predict_feature(x)

    def _predict(self, x):
        raise NotImplementedError

    def _predict_feature(self, x):
        raise NotImplementedError

    def _fit(self, train_data, val_data, time_limit=math.inf):
        raise NotImplementedError

    def _resume_fit(self, train_data, val_data, time_limit=math.inf):
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
            d.pop('_feature_net', None)
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
        d['_reporter'] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # logger
        self._logger = logging.getLogger(state.get('_name', self.__class__.__name__))
        self._logger.setLevel(logging.ERROR)
        try:
            fh = logging.FileHandler(self._log_file)
            self._logger.addHandler(fh)
        #pylint: disable=bare-except
        except:
            pass
        try:
            import mxnet as _
            net_params = state['net']
            self._init_network()
            with temporary_filename() as tfile:
                with open(tfile, 'wb') as fo:
                    fo.write(net_params)
                self.net.load_parameters(tfile, ignore_extra=True)
            trainer_state = state['trainer']
            self._init_trainer()
            with temporary_filename() as tfile:
                with open(tfile, 'wb') as fo:
                    fo.write(trainer_state)
                self.trainer.load_states(tfile)
        except ImportError:
            pass
        self._logger.setLevel(logging.INFO)
