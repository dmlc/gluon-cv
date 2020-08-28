import logging
import copy

import autogluon as ag
from autogluon.core.decorator import sample_config
from autogluon.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.task.base import BaseTask, compile_scheduler_options
from autogluon.utils import collect_params

from ... import utils as gutils
from ..estimators.base_estimator import ConfigDict, BaseEstimator
from ..estimators.ssd import SSDEstimator
from ..estimators.faster_rcnn import FasterRCNNEstimator
from ..estimators import YoloEstimator, CenterNetEstimator
from .auto_config import auto_args, config_to_nested, config_to_space


__all__ = ['ObjectDetection']

@ag.args()
def _train_object_detection(args, reporter):
    """
    args: <class 'autogluon.utils.edict.EasyDict'>
    """
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.train.seed)

    # disable auto_resume for HPO tasks
    if 'train' in args:
        args['train']['auto_resume'] = False
    else:
        args['train'] = {'auto_resume': False}

    try:
        estimator = args.estimator(args, reporter=reporter)
        # training
        estimator.fit()
    except Exception as e:
        return str(e)

    # TODO: checkpointing needs to be done in a better way
    # if args.final_fit:
    #     return {'model_params': collect_params(estimator.net)}

    return {}


class ObjectDetection(BaseTask):
    def __init__(self, config, estimator=None, logger=None):
        super(ObjectDetection, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._estimator = estimator
        self._config = ConfigDict(config)  # <class 'gluoncv.auto.estimators.base_estimator.ConfigDict'>

        if self._config.get('auto_search', True):
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

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
                "Reduce the number to {}".format(ngpus_per_trial))

        # If only time_limits is given, the scheduler starts trials until the
        # time limit is reached
        if self._config.num_trials is None and self._config.time_limits is None:
            self._config.num_trials = 2

        config.update({'num_workers': nthreads_per_trial})
        config.update({'gpus': [int(i) for i in range(ngpus_per_trial)]})
        config.update({'final_fit': False})
        config.update({'estimator': self._estimator})

        nested_config = config_to_nested(config)
        ag_space = config_to_space(nested_config)  # <class 'autogluon.core.space.Dict'>

        _train_object_detection.register_args(**ag_space)

        # self._config.scheduler_options = {
        #     'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
        #     'checkpoint': self._config.get('checkpoint', 'checkpoint/exp1.ag'),
        #     'num_trials': self._config.get('num_trials', 2),
        #     'time_out': self._config.get('time_limits', 60 * 60),
        #     'resume': (len(self._config.get('resume', '')) > 0),
        #     'visualizer': self._config.get('visualizer', 'none'),
        #     'time_attr': 'epoch',
        #     'reward_attr': 'map_reward',
        #     'dist_ip_addrs': self._config.get('dist_ip_addrs', None),
        #     'searcher': self._config.search_strategy,
        #     'search_options': self._config.get('search_options', None)}
        # if self._config.search_strategy == 'hyperband':
        #     self._config.scheduler_options.update({
        #         'searcher': 'random',
        #         'max_t': self._config.get('epochs', 50),
        #         'grace_period': self._config.grace_period if self._config.grace_period
        #         else self._config.epochs // 4})

        if self._config.grace_period is not None:
            if self._config.scheduler_options is None:
                self._config.scheduler_options = {'grace_period': self._config.grace_period}
            else:
                assert 'grace_period' not in self._config.scheduler_options, \
                    "grace_period appears both in scheduler_options and as direct argument"
                self._config.scheduler_options = copy.copy(self._config.scheduler_options)
                self._config.scheduler_options['grace_period'] = self._config.grace_period
            self._logger.warning(
                "grace_period is deprecated, use "
                "scheduler_options={'grace_period': ...} instead")
        self._config.scheduler_options = compile_scheduler_options(
            scheduler_options=self._config.scheduler_options,
            search_strategy=self._config.search_strategy,
            search_options=self._config.get('search_options', None),
            nthreads_per_trial=nthreads_per_trial,
            ngpus_per_trial=ngpus_per_trial,
            checkpoint=self._config.get('checkpoint', 'checkpoint/exp1.ag'),
            num_trials=self._config.get('num_trials', 2),
            time_out=self._config.get('time_limits', 60 * 60),
            resume=(len(self._config.get('resume', '')) > 0),
            visualizer=self._config.get('visualizer', 'none'),
            time_attr='epoch',
            reward_attr='map_reward',
            dist_ip_addrs=self._config.get('dist_ip_addrs', None),
            epochs=self._config.get('epochs', 50))

    def fit(self):
        results = self.run_fit(_train_object_detection, self._config.search_strategy,
                               self._config.scheduler_options)
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        best_config = sample_config(_train_object_detection.args, results['best_config'])
        self._logger.info('The best config: {}'.format(best_config))

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('BEST CONFIG:', best_config)
        # print('BEST CONFIG TYPE:', type(best_config))
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        estimator = self._estimator(best_config)
        # TODO: checkpointing needs to be done in a better way
        # estimator.put_parameters(results.pop('model_params'))
        return estimator
