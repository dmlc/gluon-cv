"""Auto pipeline for image classification task"""
# pylint: disable=bad-whitespace,missing-class-docstring
import time
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
from ..estimators import ImageClassificationEstimator
from .utils import config_to_nested
from .dataset import ImageClassificationDataset


__all__ = ['ImageClassification']

@dataclass
class LiteConfig:
    model : Union[type(None), str, ag.Space] = ag.Categorical('resnet18_v1b', 'mobilenetv3_small')
    lr : Union[ag.Space, float] = 1e-2
    num_trials : int = 1
    epochs : Union[ag.Space, int] = 5
    batch_size : Union[ag.Space, int] = 3  # 2 ** 3 == 8
    nthreads_per_trial : int = 32
    ngpus_per_trial : int = 0
    time_limits : int = 7 * 24 * 60 * 60  # 7 days
    search_strategy : str = 'random'
    dist_ip_addrs : Union[type(None), list, Tuple] = None

@dataclass
class DefaultConfig:
    model : Union[type(None), str, ag.Space] = ag.Categorical('resnet50_v1b', 'resnest50')
    lr : Union[ag.Space, float] = ag.Categorical(1e-2, 5e-2)
    num_trials : int = 3
    epochs : Union[ag.Space, int] = 15
    batch_size : Union[ag.Space, int] = 4  # 2 ** 4 = 16
    nthreads_per_trial : int = 128
    ngpus_per_trial : int = 8
    time_limits : int = 7 * 24 * 60 * 60  # 7 days
    search_strategy : str = 'random'
    dist_ip_addrs : Union[type(None), list, Tuple] = None

@ag.args()
def _train_image_classification(args, reporter):
    """
    Parameters
    ----------
    args: <class 'autogluon.utils.edict.EasyDict'>
    """
    # train, val data
    train_data = args.pop('train_data')
    val_data = args.pop('val_data')
    try:
        task = args.pop('task')
        dataset = args.pop('dataset')
        num_trials = args.pop('num_trials')
    except AttributeError:
        task = None
    # convert user defined config to nested form
    args = config_to_nested(args)

    tic = time.time()
    try:
        estimator_cls = args.pop('estimator', None)
        assert estimator_cls == ImageClassificationEstimator
        custom_net = args.pop('custom_net', None)
        custom_optimizer = args.pop('custom_optimizer', None)
        estimator = estimator_cls(args, reporter=reporter,
                                  net=custom_net, optimizer=custom_optimizer)
        # training
        result = estimator.fit(train_data=train_data, val_data=val_data)
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
                'time': time.time() - tic, 'train_acc': -1, 'valid_acc': -1}

    # TODO: checkpointing needs to be done in a better way
    # unique_checkpoint = 'train_image_classification_' + str(uuid.uuid4()) + '.pkl'
    # estimator.save(unique_checkpoint)
    result.update({'model_checkpoint': pickle.dumps(estimator)})
    return result


class ImageClassification(BaseTask):
    """Whole Image Classification general task.

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.
    net : mx.gluon.Block
        The custom network. If defined, the model name in config will be ignored so your
        custom network will be used for training rather than pulling it from model zoo.
    """
    Dataset = ImageClassificationDataset

    def __init__(self, config=None, estimator=None, logger=None):
        super(ImageClassification, self).__init__()
        self._fit_summary = {}
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)


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


        # additional configs
        config['num_workers'] = nthreads_per_trial
        config['gpus'] = [int(i) for i in range(ngpus_per_trial)]
        # if config['gpus']:
        #     config['batch_size'] = config.get('batch_size', 8) * len(config['gpus'])
        #     self._logger.info('Increase batch size to %d based on the number of gpus %d',
        #                       config['batch_size'], len(config['gpus']))
        config['seed'] = config.get('seed', np.random.randint(32,767))
        self._config = config

        # scheduler options
        self.search_strategy = config.get('search_strategy', 'random')
        self.search_options = config.get('search_options', None)
        if self.search_options:
            self.search_options.update({'debug_log': True})
        else:
            self.search_options = {'debug_log': True}
        self.scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': config.get('checkpoint', 'checkpoint/exp1.ag'),
            'num_trials': config.get('num_trials', 2),
            'time_out': config.get('time_limits', 60 * 60),
            'resume': (len(config.get('resume', '')) > 0),
            'visualizer': config.get('visualizer', 'none'),
            'time_attr': 'epoch',
            'reward_attr': 'acc_reward',
            'dist_ip_addrs': config.get('dist_ip_addrs', None),
            'searcher': self.search_strategy,
            'search_options': self.search_options}
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

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None):
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
        # estimator = self._config.get('estimator', None)
        # if estimator is None:
        #     estimator = [ImageClassificationEstimator]
        # elif isinstance(estimator, (tuple, list)):
        #     pass
        # else:
        #     assert issubclass(estimator, BaseEstimator)
        #     estimator = [estimator]
        # self._config['estimator'] = ag.Categorical(*estimator)

        estimator = self._config.get('estimator', None)
        if estimator is None:
            estimator = [ImageClassificationEstimator]
        else:
            if isinstance(estimator, ag.Space):
                estimator = estimator.data
            elif isinstance(estimator, str):
                estimator = [estimator]
            for i, e in enumerate(estimator):
                if e == 'img_cls':
                    estimator[i] = ImageClassificationEstimator
                else:
                    estimator.pop(e)
        self._config['estimator'] = ag.Categorical(*estimator)

        # register args
        config = self._config.copy()
        config['train_data'] = train_data
        config['val_data'] = val_data
        _train_image_classification.register_args(**config)

        start_time = time.time()
        self._fit_summary = {}
        if config.get('num_trials', 1) < 2:
            rand_config = RandomSearcher(_train_image_classification.cs).get_config()
            self._logger.info("Starting fit without HPO")
            results = _train_image_classification(_train_image_classification.args, rand_config)
            best_config = sample_config(_train_image_classification.args, rand_config)
            best_config.pop('train_data', None)
            best_config.pop('val_data', None)
            self._fit_summary.update({'train_acc': results.get('train_acc', -1),
                                      'valid_acc': results.get('valid_acc', -1),
                                      'total_time': results.get('time', time.time() - start_time),
                                      'best_config': best_config})
        else:
            self._logger.info("Starting HPO experiments")
            results = self.run_fit(_train_image_classification, self.search_strategy,
                                   self.scheduler_options)
        end_time = time.time()
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        self._logger.info("total runtime is %.2f s", end_time - start_time)
        if config.get('num_trials', 1) > 1:
            best_config = sample_config(_train_image_classification.args, results['best_config'])
            # convert best config to nested form
            best_config = config_to_nested(best_config)
            best_config.pop('train_data', None)
            best_config.pop('val_data', None)
            self._fit_summary.update({'train_acc': results.get('train_acc', -1),
                                      'valid_acc': results.get('valid_acc', results.get('best_reward', -1)),
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

    @classmethod
    def load(cls, filename):
        obj = BaseEstimator.load(filename)
        # make sure not accidentally loading e.g. classification model
        assert isinstance(obj, ImageClassificationEstimator)
        return obj
