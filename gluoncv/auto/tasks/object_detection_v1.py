import logging

import autogluon as ag
from autogluon.core.decorator import sample_config
from autogluon.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.task import BaseTask
from autogluon.utils import collect_params

from ... import utils as gutils
from ..estimators.base_estimator import ConfigDict, BaseEstimator
from ..estimators.ssd import SSDEstimator
from ..estimators.faster_rcnn import FasterRCNNEstimator
from ..estimators import YoloEstimator, CenterNetEstimator
from .auto_config import auto_args


__all__ = ['ObjectDetection']

@ag.args()
def _train_object_detection(args, reporter):
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # disable auto_resume for HPO tasks
    if 'train' in config:
        config['train']['auto_resume'] = False
    else:
        config['train'] = {'auto_resume': False}

    try:
        estimator = args.estimator(config, reporter=reporter)
        # training
        estimator.fit()
    except Exception as e:
        return str(e)

    # (TODO): checkpointing needs to be done in a better way
    # if args.final_fit:
    #     return {'model_params': collect_params(estimator.net)}

    return {}


class ObjectDetection(BaseTask):
    def __init__(self, config, estimator=None, logger=None):
        super(ObjectDetection, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._estimator = estimator
        # self._config = config
        self._config = ConfigDict(config)

        cpu_count = get_cpu_count()
        gpu_count = get_gpu_count()
        nthreads_per_trial = self._config.get('nthreads_per_trial', cpu_count)
        if nthreads_per_trial > cpu_count:
            nthreads_per_trial = cpu_count
        ngpus_per_trial = self._config.get('ngpus_per_trial', gpu_count)
        if ngpus_per_trial > gpu_count:
            ngpus_per_trial = gpu_count
            self._logger.warning(
                "The number of requested GPUs is greater than the number of available GPUs."
                "Reduce the number to {}".format(ngpus_per_trial))

        if estimator is None:
            estimator = [SSDEstimator, FasterRCNNEstimator, YoloEstimator, CenterNetEstimator]
        elif isinstance(estimator, (tuple, list)):
            pass
        else:
            assert issubclass(estimator, BaseEstimator)
            estimator = [estimator]
        config['estimator'] = ag.space.Categorical(*estimator)

        # automatically merge search configs according to user specified values
        args = auto_args(estimator, config)
        args.num_workers = nthreads_per_trial
        args.gpus = [int(i) for i in range(ngpus_per_trial)]

        _train_object_detection.register_args(**args)

        self._config.search_strategy = self._config.get('search_strategy', 'random')
        self._config.scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': self._config.get('checkpoint', 'checkpoint/exp1.ag'),
            'num_trials': self._config.get('num_trials', 3),
            'time_out': self._config.get('time_limits', 60 * 60),
            'resume': (len(self._config.get('resume', '')) > 0),
            'visualizer': self._config.get('visualizer', 'none'),
            'time_attr': 'epoch',
            'reward_attr': 'map_reward',
            'dist_ip_addrs': self._config.get('dist_ip_addrs', []),
            'searcher': self._config.search_strategy,
            'search_options': self._config.get('search_options', {}),
        }
        if self._config.search_strategy == 'hyperband':
            self._config.scheduler_options.update({
                'searcher': 'random',
                'max_t': self._config.epochs,
                'grace_period': self._config.get('grace_period', self._config.epochs // 4)
            })

    def fit(self):
        results = self.run_fit(_train_object_detection, self._config.search_strategy,
                               self._config.scheduler_options)
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        best_config = sample_config(_train_object_detection.args, results['best_config'])
        self._logger.info('The best config: {}'.format(best_config))

        estimator = self._estimator(best_config)
        # (TODO): checkpointing needs to be done in a better way
        # estimator.put_parameters(results.pop('model_params'))
        return estimator
