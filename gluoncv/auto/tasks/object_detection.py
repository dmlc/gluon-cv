"""Auto pipeline for object detection task"""
import time
import logging
import uuid

import numpy as np
import pandas as pd
import autogluon.core as ag
from autogluon.core.decorator import sample_config
from autogluon.core.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.core.task.base import BaseTask

from ... import utils as gutils
from ..estimators.base_estimator import ConfigDict, BaseEstimator
from ..estimators import SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator
from .utils import auto_suggest, config_to_nested
from .dataset import ObjectDetectionDataset


__all__ = ['ObjectDetection']

@ag.args()
def _train_object_detection(args, reporter):
    """
    Parameters
    ----------
    args: <class 'autogluon.utils.edict.EasyDict'>
    """
    # convert user defined config to nested form
    args = config_to_nested(args)

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args['train']['seed'])

    # disable auto_resume for HPO tasks
    if 'train' in args:
        args['train']['auto_resume'] = False
    else:
        args['train'] = {'auto_resume': False}

    # train, val data
    train_data = args.pop('train_data')
    val_data = args.pop('val_data')
    args.pop('meta_arch', None)

    try:
        estimator_cls = args.pop('estimator', None)
        estimator = estimator_cls(args, reporter=reporter)
        # training
        estimator.fit(train_data=train_data, val_data=val_data)
    # pylint: disable=broad-except
    except Exception as e:
        return {'stacktrace': str(e)}

    # TODO: checkpointing needs to be done in a better way
    unique_checkpoint = str(uuid.uuid4())
    estimator.save(unique_checkpoint)
    return {'model_checkpoint': unique_checkpoint}


class ObjectDetection(BaseTask):
    """Object Detection general task.

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.

    """
    Dataset = ObjectDetectionDataset

    def __init__(self, config, logger=None):
        super(ObjectDetection, self).__init__()
        self._config = ConfigDict(config)
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.info("Starting HPO experiments")

        # cpu and gpu setting
        cpu_count = get_cpu_count()
        nthreads_per_trial = self._config.get('nthreads_per_trial', cpu_count)
        if nthreads_per_trial > cpu_count:
            nthreads_per_trial = cpu_count
        gpu_count = get_gpu_count()
        ngpus_per_trial = self._config.get('ngpus_per_trial', gpu_count)
        if ngpus_per_trial > gpu_count:
            ngpus_per_trial = gpu_count
            self._logger.warning(
                "The number of requested GPUs is greater than the number of available GPUs."
                "Reduce the number to %d", ngpus_per_trial)

        # additional configs
        config['num_workers'] = nthreads_per_trial
        config['gpus'] = [int(i) for i in range(ngpus_per_trial)]
        config['seed'] = self._config.get('seed', 233)
        config['final_fit'] = False
        # config['logger'] = self._logger


        # automatically merge search configs according to user specified values
        # args = auto_args(config, estimator)

        # register args for HPO
        # _train_object_detection.register_args(**args)
        # _train_object_detection.register_args(**config)
        self._train_config = config

        # scheduler options
        self._config.search_strategy = self._config.get('search_strategy', 'random')
        self._config.scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': self._config.get('checkpoint', 'checkpoint/exp1.ag'),
            'num_trials': self._config.get('num_trials', 2),
            'time_out': self._config.get('time_limits', 60 * 60),
            'resume': (len(self._config.get('resume', '')) > 0),
            'visualizer': self._config.get('visualizer', 'none'),
            'time_attr': 'epoch',
            'reward_attr': 'map_reward',
            'dist_ip_addrs': self._config.get('dist_ip_addrs', None),
            'searcher': self._config.search_strategy,
            'search_options': self._config.get('search_options', None)}
        if self._config.search_strategy == 'hyperband':
            self._config.scheduler_options.update({
                'searcher': 'random',
                'max_t': self._config.get('epochs', 50),
                'grace_period': self._config.get('grace_period', self._config.epochs // 4)})

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None):
        """Fit auto estimator given the input data.

        Returns
        -------
        Estimator
            The estimator obtained by training on the specified dataset.

        """
        # split train/val before HPO to make fair comparisons
        if not isinstance(train_data, pd.DataFrame):
            assert val_data is not None, \
                "Please provide `val_data` as we do not know how to split `train_data` of type: \
                {}".format(type(train_data))

        if not val_data:
            assert 0 <= train_size <= 1.0
            if random_state:
                np.random.seed(random_state)
            split_mask = np.random.rand(len(train_data)) < train_size
            train = train_data[split_mask]
            val = train_data[~split_mask]
            self._logger.info('Randomly split train_data into train[%d]/validation[%d] splits.',
                              len(train), len(val))
            train_data, val_data = train, val

        # automatically suggest some hyperparameters based on the dataset statistics
        estimator = self._train_config.get('estimator', None)
        self._train_config['train_dataset'] = train_data
        if self._config.get('auto_suggest', True):
            auto_suggest(self._train_config, estimator, self._logger)
        else:
            if estimator is None:
                estimator = [SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator]
            elif isinstance(estimator, (tuple, list)):
                pass
            else:
                assert issubclass(estimator, BaseEstimator)
                estimator = [estimator]
            self._train_config['estimator'] = ag.Categorical(*estimator)
        self._train_config.pop('train_dataset')

        # register args
        config = self._train_config.copy()
        config['train_data'] = train_data
        config['val_data'] = val_data
        _train_object_detection.register_args(**config)

        start_time = time.time()

        results = self.run_fit(_train_object_detection, self._config.search_strategy,
                               self._config.scheduler_options)
        end_time = time.time()
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        self._logger.info("total runtime is %.2f s", end_time - start_time)
        best_config = sample_config(_train_object_detection.args, results['best_config'])
        # convert best config to nested form
        best_config = config_to_nested(best_config)
        best_config.pop('train_data')
        best_config.pop('val_data')
        self._logger.info('The best config: %s', str(best_config))

        # estimator = best_config['estimator'](best_config)
        # TODO: checkpointing needs to be done in a better way
        model_checkpoint = results.get('model_checkpoint', None)
        if model_checkpoint is None:
            msg = results.get('stacktrace', '')
            raise RuntimeError(f'Unexpected error happened during fit: {msg}')
        estimator = self.load(results['model_checkpoint'])
        return estimator

    @classmethod
    def load(cls, filename):
        obj = BaseEstimator.load(filename)
        # make sure not accidentally loading e.g. classification model
        # pylint: disable=unidiomatic-typecheck
        assert type(obj) in (SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator)
        return obj
