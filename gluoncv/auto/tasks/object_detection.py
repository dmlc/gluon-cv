"""Auto pipeline for object detection task"""
# pylint: disable=bad-whitespace,missing-class-docstring
import time
import math
import copy
import logging
import pprint
import json
import pickle
from typing import Union, Tuple

from autocfg import dataclass
import numpy as np
import pandas as pd
import autogluon.core as ag
from autogluon.core.decorator import sample_config
from autogluon.core.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.core.task.base import BaseTask
from autogluon.core.searcher import RandomSearcher

from ..estimators.base_estimator import BaseEstimator
from ..estimators import SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator
from .utils import auto_suggest, config_to_nested
from ..data.dataset import ObjectDetectionDataset

__all__ = ['ObjectDetection']

@dataclass
class LiteConfig:
    transfer : Union[type(None), str, ag.Space] = ag.Categorical('ssd_512_mobilenet1.0_coco', 'yolo3_mobilenet1.0_coco')
    lr : Union[ag.Space, float] = 1e-3
    num_trials : int = 1
    epochs : Union[ag.Space, int] = 5
    nthreads_per_trial : int = 32
    ngpus_per_trial : int = 0
    time_limits : int = 7 * 24 * 60 * 60  # 7 days
    search_strategy : str = 'random'
    dist_ip_addrs : Union[type(None), list, Tuple] = None

@dataclass
class DefaultConfig:
    transfer : Union[type(None), str, ag.Space] = ag.Categorical('ssd_512_resnet50_v1_coco',
                                                                 'yolo3_darknet53_coco',
                                                                 'faster_rcnn_resnet50_v1b_coco',
                                                                 'center_net_resnet50_v1b_coco')
    lr : Union[ag.Space, float] = ag.Categorical(1e-3, 5e-3)
    num_trials : int = 3
    epochs : Union[ag.Space, int] = 10
    nthreads_per_trial : int = 128
    ngpus_per_trial : int = 8
    time_limits : int = 7 * 24 * 60 * 60  # 7 days
    search_strategy : str = 'random'
    dist_ip_addrs : Union[type(None), list, Tuple] = None


@ag.args()
def _train_object_detection(args, reporter):
    """
    Parameters
    ----------
    args: <class 'autogluon.utils.edict.EasyDict'>
    """
    tic = time.time()
    # train, val data
    train_data = args.pop('train_data')
    val_data = args.pop('val_data')
    # wall clock tick limit
    wall_clock_tick = args.pop('wall_clock_tick')
    # exponential batch size for Int() space batch sizes
    try:
        exp_batch_size = args.pop('exp_batch_size')
    except AttributeError:
        exp_batch_size = False
    if exp_batch_size and 'batch_size' in args:
        args['batch_size'] = 2 ** args['batch_size']
    try:
        task = args.pop('task')
        dataset = args.pop('dataset')
        num_trials = args.pop('num_trials')
    except AttributeError:
        task = None
    # convert user defined config to nested form
    args = config_to_nested(args)

    try:
        estimator_cls = args.pop('estimator', None)
        if estimator_cls == FasterRCNNEstimator:
            # safe guard if too many GT in dataset
            train_dataset = train_data.to_mxnet()
            max_gt_count = max([y[1].shape[0] for y in train_dataset]) + 20
            args['faster_rcnn']['max_num_gt'] = max_gt_count
        estimator = estimator_cls(args, reporter=reporter)
        # training
        result = estimator.fit(train_data=train_data, val_data=val_data, time_limit=wall_clock_tick-tic)
        # save config and result
        if task is not None:
            trial_log = {}
            trial_log.update(args)
            trial_log.update(result)
            json_str = json.dumps(trial_log)
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            json_file_name = task + '_dataset-' + dataset + '_trials-' + str(num_trials) + '_' + time_str + '.json'
            with open(json_file_name, 'w') as json_file:
                json_file.write(json_str)
            logging.info('Config and result in this trial have been saved to %s.', json_file_name)
    # pylint: disable=bare-except
    except:
        import traceback
        return {'traceback': traceback.format_exc(), 'args': str(args),
                'time': time.time() - tic, 'train_map': -1, 'valid_map': -1}

    # TODO: checkpointing needs to be done in a better way
    # unique_checkpoint = 'train_object_detection_' + str(uuid.uuid4()) + '.pkl'
    # estimator.save(unique_checkpoint)
    result.update({'model_checkpoint': pickle.dumps(estimator)})
    return result

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
        self._fit_summary = {}
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._fit_summary = {}
        self._results = {}

        # cpu and gpu setting
        cpu_count = get_cpu_count()
        gpu_count = get_gpu_count()

        # default settings
        if not config:
            if gpu_count < 1:
                self._logger.info('No GPU detected/allowed, using most conservative search space.')
                config = LiteConfig()
            else:
                config = DefaultConfig()
            config = config.asdict()
        else:
            if not config.get('dist_ip_addrs', None):
                ngpus_per_trial = config.get('ngpus_per_trial', gpu_count)
                if ngpus_per_trial > gpu_count:
                    ngpus_per_trial = gpu_count
                if ngpus_per_trial < 1:
                    self._logger.info('No GPU detected/allowed, using most conservative search space.')
                    default_config = LiteConfig()
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
            raise ValueError('Please specify `nthreads_per_trial` and `ngpus_per_trial` '
                             'given that dist workers are available')

        # fix estimator-transfer relationship
        estimator = config.get('estimator', None)
        transfer = config.get('transfer', None)
        if estimator is not None and transfer is not None:
            if isinstance(estimator, ag.Space):
                estimator = estimator.data
            elif isinstance(estimator, str):
                estimator = [estimator]
            if isinstance(transfer, ag.Space):
                transfer = transfer.data
            elif isinstance(transfer, str):
                transfer = [transfer]

            valid_transfer = []
            for e in estimator:
                for t in transfer:
                    if e in t:
                        valid_transfer.append(t)

            if not valid_transfer:
                raise ValueError(f'No matching `transfer` model for {estimator}')
            if len(valid_transfer) == 1:
                config['transfer'] = valid_transfer[0]
            else:
                config['transfer'] = ag.Categorical(*valid_transfer)

        # additional configs
        config['num_workers'] = nthreads_per_trial
        config['gpus'] = [int(i) for i in range(ngpus_per_trial)]
        config['seed'] = config.get('seed', np.random.randint(32,767))
        self._config = config

        # scheduler options
        self.search_strategy = config.get('search_strategy', 'random')
        self.search_options = config.get('search_options', {})
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
            'search_options': self.search_options,
            'max_reward': config.get('max_reward', 0.9)}
        if self.search_strategy == 'hyperband':
            self.scheduler_options.update({
                'searcher': 'random',
                'max_t': config.get('epochs', 50),
                'grace_period': config.get('grace_period', config.get('epochs', 50) // 4)})
        elif self.search_strategy == 'bayesopt_hyperband':
            self.scheduler_options.update({
                'searcher': 'bayesopt',
                'max_t': config.get('epochs', 50),
                'grace_period': config.get('grace_period', config.get('epochs', 50) // 4)})

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None, time_limit=None):
        """Fit auto estimator given the input data.

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
        time_limit : int, default is None
            The wall clock time limit(second) for fit process, if `None`, time limit is not enforced.
            If `fit` takes longer than `time_limit`, the process will terminate early and return the
            model prematurally.
            Due to callbacks and additional validation functions, the `time_limit` may not be very precise
            (few minutes allowance), but you can use it to safe-guard a very long training session.

        Returns
        -------
        Estimator
            The estimator obtained by training on the specified dataset.

        """
        if time_limit is None:
            time_limit = math.inf
        elif not isinstance(time_limit, int):
            raise TypeError(f'Invalid type `time_limit={time_limit}`, int or None expected')
        wall_clock_tick = time.time() + time_limit
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
        estimator = self._config.get('estimator', None)
        transfer = self._config.get('transfer', None)
        if not transfer:
            self._config['train_dataset'] = train_data
            auto_suggest(self._config, estimator, self._logger)
            self._config.pop('train_dataset')

        # register args
        config = self._config.copy()
        config['train_data'] = train_data
        config['val_data'] = val_data
        config['wall_clock_tick'] = wall_clock_tick
        _train_object_detection.register_args(**config)

        start_time = time.time()
        self._fit_summary = {}
        self._results = {}
        if config.get('num_trials', 1) < 2:
            rand_config = RandomSearcher(_train_object_detection.cs).get_config()
            self._logger.info("Starting fit without HPO")
            results = _train_object_detection(_train_object_detection.args, rand_config)
            best_config = sample_config(_train_object_detection.args, rand_config)
            best_config.pop('train_data', None)
            best_config.pop('val_data', None)
            self._fit_summary.update({'train_map': results.get('train_map', -1),
                                      'valid_map': results.get('valid_map', -1),
                                      'total_time': results.get('time', time.time() - start_time),
                                      'best_config': best_config})
        else:
            self._logger.info("Starting HPO experiments")
            results = self.run_fit(_train_object_detection, self.search_strategy,
                                   self.scheduler_options)
            if isinstance(results, dict):
                ks = ('best_reward', 'best_config', 'total_time', 'config_history', 'reward_attr')
                self._results.update({k: v for k, v in results.items() if k in ks})
        end_time = time.time()
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        self._logger.info("total runtime is %.2f s", end_time - start_time)
        if config.get('num_trials', 1) > 1:
            best_config = sample_config(_train_object_detection.args, results['best_config'])
            # convert best config to nested form
            best_config = config_to_nested(best_config)
            best_config.pop('train_data', None)
            best_config.pop('val_data', None)
            self._fit_summary.update({'train_map': results.get('train_map', -1),
                                      'valid_map': results.get('valid_map', results.get('best_reward', -1)),
                                      'total_time': results.get('total_time', time.time() - start_time),
                                      'best_config': best_config})
        self._logger.info(pprint.pformat(self._fit_summary, indent=2))

        # TODO: checkpointing needs to be done in a better way
        model_checkpoint = results.get('model_checkpoint', None)
        if model_checkpoint is None:
            raise RuntimeError(f'Unexpected error happened during fit: {pprint.pformat(results, indent=2)}')
        estimator = pickle.loads(results['model_checkpoint'])
        return estimator

    def fit_summary(self):
        return copy.copy(self._fit_summary)

    def fit_history(self):
        return copy.copy(self._results)

    @classmethod
    def load(cls, filename):
        obj = BaseEstimator.load(filename)
        # make sure not accidentally loading e.g. classification model
        # pylint: disable=unidiomatic-typecheck
        assert type(obj) in (SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator)
        return obj
