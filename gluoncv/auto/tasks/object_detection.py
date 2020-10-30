"""Auto pipeline for object detection task"""
# pylint: disable=bad-whitespace
import time
import logging
import uuid
from typing import Union, Tuple

from autocfg import dataclass, field
import numpy as np
import pandas as pd
import autogluon.core as ag
from autogluon.core.decorator import sample_config
from autogluon.core.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.core.task.base import BaseTask

from ... import utils as gutils
from .utils import ConfigDict
from ..estimators.base_estimator import BaseEstimator
from ..estimators import SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator
from .utils import auto_suggest, config_to_nested
from .dataset import ObjectDetectionDataset

__all__ = ['ObjectDetection']

@dataclass
class LightConfig:
    transfer : Union[str, ag.Space] = ag.Categorical('center_net_resnet18_v1b_coco', 'ssd_512_mobilenet1.0_coco')
    lr : Union[ag.Space, float] = 1e-2
    num_trials : int = 2
    epochs : int = 10
    nthreads_per_trial : int = 32
    ngpus_per_trial : int = 0
    time_limits : int = 3600
    search_strategy : str = 'random'
    dist_ip_addrs : Union[None, list, Tuple] = None

@dataclass
class DefaultConfig:
    transfer : Union[ag.Space, str] = ag.Categorical('center_net_resnet50_v1b_coco', 'yolo3_darknet53_coco')
    lr : Union[ag.Space, float] = ag.Categorical(1e-2, 5e-3)
    num_trials : int = 3
    epochs : int = 15
    nthreads_per_trial : int = 128
    ngpus_per_trial : int = 8
    time_limits : int = 3600
    search_strategy : str = 'random'
    dist_ip_addrs : Union[None, list, Tuple] = None


@ag.args()
def _train_object_detection(args, reporter):
    """
    Parameters
    ----------
    args: <class 'autogluon.utils.edict.EasyDict'>
    """
    # train, val data
    train_data = args.pop('train_data')
    val_data = args.pop('val_data')
    # convert user defined config to nested form
    args = config_to_nested(args)

    try:
        estimator_cls = args.pop('estimator', None)
        estimator = estimator_cls(args, reporter=reporter)
        # training
        estimator.fit(train_data=train_data, val_data=val_data)
    # pylint: disable=broad-except
    except Exception as e:
        return {'stacktrace': str(e), 'args': str(args)}

    # TODO: checkpointing needs to be done in a better way
    unique_checkpoint = 'train_object_detection_' + str(uuid.uuid4())
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

    def __init__(self, config=None, logger=None):
        super(ObjectDetection, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        # cpu and gpu setting
        cpu_count = get_cpu_count()
        gpu_count = get_gpu_count()

        # default settings
        if not config:
            if gpu_count < 1:
                self._logger.info('No GPU detected/allowed, using most conservative search space.')
                config = LightConfig()
            else:
                config = DefaultConfig()
            config = config.asdict()
        else:
            if not config.get('dist_ip_addrs', None):
                ngpus_per_trial = config.get('ngpus_per_trial', gpu_count)
                if ngpus_per_trial < 1:
                    self._logger.info('No GPU detected/allowed, using most conservative search space.')
                    default_config = LightConfig()
                else:
                    default_config = DefaultConfig()
                config = default_config.merge(config, allow_new_key=True).asdict()

        # adjust cpu/gpu resources
        if not config.get('dist_ip_addrs', None):
            nthreads_per_trial = config.get('nthreads_per_trial', cpu_count)
            if nthreads_per_trial > cpu_count:
                nthreads_per_trial = cpu_count
            ngpus_per_trial = config.get('ngpus_per_trial', gpu_count)
            if ngpus_per_trial > gpu_count:
                ngpus_per_trial = gpu_count
                self._logger.warning(
                    "The number of requested GPUs is greater than the number of available GPUs."
                    "Reduce the number to %d", ngpus_per_trial)
        else:
            raise ValueError('Please specify `nthreads_per_trial` and `ngpus_per_trial` given that dist workers are available')

        # additional configs
        config['num_workers'] = nthreads_per_trial
        config['gpus'] = [int(i) for i in range(ngpus_per_trial)]
        config['seed'] = config.get('seed', np.random.randint(10000))
        config['final_fit'] = False
        self._config = config

        # scheduler options
        self.search_strategy = config.get('search_strategy', 'random')
        self.scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': config.get('checkpoint', 'checkpoint/exp1.ag'),
            'num_trials': config.get('num_trials', 2),
            'time_out': config.get('time_limits', 60 * 60),
            'resume': (len(config.get('resume', '')) > 0),
            'visualizer': config.get('visualizer', 'none'),
            'time_attr': 'epoch',
            'reward_attr': 'map_reward',
            'dist_ip_addrs': config.get('dist_ip_addrs', None),
            'searcher': self.search_strategy,
            'search_options': config.get('search_options', None)}
        if self.search_strategy == 'hyperband':
            self.scheduler_options.update({
                'searcher': 'random',
                'max_t': config.get('epochs', 50),
                'grace_period': config.get('grace_period', config.get('epochs', 50) // 4)})

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

        if val_data is None:
            assert 0 <= train_size <= 1.0
            if random_state:
                np.random.seed(random_state)
            split_mask = np.random.rand(len(train_data)) < train_size
            train = train_data[split_mask]
            val = train_data[~split_mask]
            self._logger.info('Randomly split train_data into train[%d]/validation[%d] splits.',
                              len(train), len(val))
            train_data, val_data = train, val

        # automatically suggest some hyperparameters based on the dataset statistics(experimental)
        if not self._config.get('transfer', None):
            estimator = self._config.get('estimator', None)
            if estimator is None:
                estimator = [SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator]
            self._config['estimator'] = ag.Categorical(*estimator)
            self._config['train_dataset'] = train_data
            if self._config.get('auto_suggest', True):
                auto_suggest(self._config, estimator, self._logger)
            self._config.pop('train_dataset')

        # register args
        config = self._config.copy()
        config['train_data'] = train_data
        config['val_data'] = val_data
        _train_object_detection.register_args(**config)

        start_time = time.time()
        self._logger.info("Starting HPO experiments")

        results = self.run_fit(_train_object_detection, self.search_strategy,
                               self.scheduler_options)
        end_time = time.time()
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        self._logger.info("total runtime is %.2f s", end_time - start_time)
        best_config = sample_config(_train_object_detection.args, results['best_config'])
        # convert best config to nested form
        best_config = config_to_nested(best_config)
        best_config.pop('train_data', None)
        best_config.pop('val_data', None)
        self._logger.info('The best config: %s', str(best_config))

        # estimator = best_config['estimator'](best_config)
        # TODO: checkpointing needs to be done in a better way
        model_checkpoint = results.get('model_checkpoint', None)
        if model_checkpoint is None:
            msg = results.get('stacktrace', '')
            args = results.get('args', '')
            raise RuntimeError(f'Unexpected error happened during fit: {msg} with {args}')
        estimator = self.load(results['model_checkpoint'])
        return estimator

    @classmethod
    def load(cls, filename):
        obj = BaseEstimator.load(filename)
        # make sure not accidentally loading e.g. classification model
        # pylint: disable=unidiomatic-typecheck
        assert type(obj) in (SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator)
        return obj
