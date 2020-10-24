"""Auto pipeline for image classification task"""
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
from ..estimators import ImageClassificationEstimator
from .utils import auto_suggest, config_to_nested
from .dataset import ImageClassificationDataset


__all__ = ['ImageClassification']

@ag.args()
def _train_image_classification(args, reporter):
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

    try:
        estimator = args['estimator'](args, reporter=reporter)
        # training
        estimator.fit(train_data=train_data, val_data=val_data)
    # pylint: disable=broad-except
    except Exception as e:
        return {'stacktrace': str(e)}

    # TODO: checkpointing needs to be done in a better way
    unique_checkpoint = str(uuid.uuid4())
    estimator.save(unique_checkpoint)
    return {'model_checkpoint': unique_checkpoint}


class ImageClassification(BaseTask):
    """Whole Image Classification general task.

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.

    """
    Dataset = ImageClassificationDataset

    def __init__(self, config, estimator=None, logger=None):
        super(ImageClassification, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._config = ConfigDict(config)

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

        # automatically merge search configs according to user specified values
        # args = auto_args(config, estimator)

        # register args for HPO
        # _train_object_detection.register_args(**args)
        # _train_image_classification.register_args(**config)
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
            'reward_attr': 'acc_reward',
            'dist_ip_addrs': self._config.get('dist_ip_addrs', None),
            'searcher': self._config.search_strategy,
            'search_options': self._config.get('search_options', None)}
        if self._config.search_strategy == 'hyperband':
            self._config.scheduler_options.update({
                'searcher': 'random',
                'max_t': self._config.get('epochs', 50),
                'grace_period': self._config.get('grace_period', self._config.epochs // 4)})

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None):
        """Fit auto estimator given the input data .

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

        if estimator is None:
            estimator = [ImageClassificationEstimator]
        elif isinstance(estimator, (tuple, list)):
            pass
        else:
            assert issubclass(estimator, BaseEstimator)
            estimator = [estimator]
        self._train_config['estimator'] = ag.Categorical(*estimator)

        # register args
        config = self._train_config.copy()
        config['train_data'] = train_data
        config['val_data'] = val_data
        _train_image_classification.register_args(**config)

        results = self.run_fit(_train_image_classification, self._config.search_strategy,
                               self._config.scheduler_options)
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        best_config = sample_config(_train_image_classification.args, results['best_config'])
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
        assert isinstance(obj, ImageClassificationEstimator)
        return obj
