import logging
import copy

import autogluon as ag
from autogluon.core.decorator import sample_config
from autogluon.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.task.base import BaseTask, compile_scheduler_options
from autogluon.utils import collect_params

from ... import utils as gutils
from ..estimators.base_estimator import ConfigDict
from .auto_config import config_to_nested, config_to_dict, config_to_space


__all__ = ['ObjectDetection']

@ag.args()
def _train_object_detection(args, reporter):
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print('ARGS:', args)
    # print('ARGS TYPE:', type(args))  # <class 'autogluon.utils.edict.EasyDict'>
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # for k, v in args.items():
    #     print('key:', k)
    #     print('key type:', type(k))  # <class 'str'>
    #     print('value:', v)
    #     print('value type:', type(v))  # <class 'str'>
    #     break

    # fix seed for mxnet, numpy and python builtin random generator.
    # gutils.random.seed(args.seed)
    gutils.random.seed(args.train.seed)

    # config = config_to_nested(args)
    config = config_to_dict(args)

    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print('CONFIG DICT:', config)
    # print('CONFIG DICT TYPE:', type(config))  # <class 'dict'>
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # disable auto_resume for HPO tasks
    if 'train' in config:
        config['train']['auto_resume'] = False
    else:
        config['train'] = {'auto_resume': False}

    try:
        if args.meta_arch == 'ssd' or 'faster_rcnn' or 'yolo3' or 'center_net':
            estimator = args.estimator(config, reporter=reporter)
        else:
            raise NotImplementedError('%s' % args.meta_arch)
        # training
        estimator.fit()
    except Exception as e:
        return str(e)

    if args.final_fit:
        return {'model_params': collect_params(estimator.net)}

    return {}


class ObjectDetection(BaseTask):
    def __init__(self, config, estimator, logger=None):
        super(ObjectDetection, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._estimator = estimator
        # self._config = config
        self._config = ConfigDict(config)

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('SEARCH SPACE:', self._config.lr)
        # print('SEARCH SPACE TYPE:', type(self._config.lr))  # <class 'autogluon.core.space.Categorical'>
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('INPUT CONFIG:', config)
        # print('INPUT CONFIG TYPE:', type(config))  # <class 'dict'>
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('CLASS CONFIG:', self._config)
        # print('CLASS CONFIG TYPE:', type(self._config))  # <class 'gluoncv.auto.estimators.base_estimator.ConfigDict'>
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        assert self._config.search_strategy not in {'bayesopt', 'bayesopt_hyperband'}, \
            "search_strategy == 'bayesopt' or 'bayesopt_hyperband' not yet supported"
        if self._config.auto_search:
            # The strategies can be injected here, for example: automatic suggest some hps
            # based on the dataset statistics
            pass

        self._config.nthreads_per_trial = get_cpu_count() if self._config.nthreads_per_trial > get_cpu_count() \
            else self._config.nthreads_per_trial
        if self._config.ngpus_per_trial > get_gpu_count():
            self._logger.warning(
                "The number of requested GPUs is greater than the number of available GPUs.")
        self._config.ngpus_per_trial = get_gpu_count() if self._config.ngpus_per_trial > get_gpu_count() \
            else self._config.ngpus_per_trial

        # If only time_limits is given, the scheduler starts trials until the
        # time limit is reached
        if self._config.num_trials is None and self._config.time_limits is None:
            self._config.num_trials = 2

        config.update({'final_fit': False})
        config.update({'estimator': self._estimator})
        nested_config = config_to_nested(config)

        ag_space = config_to_space(config)
        ag_space_nested = config_to_space(nested_config)

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('NESTED CONFIG:', nested_config)
        # print('NESTED CONFIG TYPE:', type(nested_config))  # <class 'dict'>
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('CONFIG SPACE:', ag_space)
        # print('CONFIG SPACE TYPE:', type(ag_space))  # <class 'autogluon.core.space.Dict'>
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        _train_object_detection.register_args(**ag_space_nested)

        # self._config.scheduler_options = {
        #     'resource': {'num_cpus': self._config.nthreads_per_trial,
        #                  'num_gpus': self._config.ngpus_per_trial},
        #     'checkpoint': self._config.checkpoint,
        #     'num_trials': self._config.num_trials,
        #     'time_out': self._config.time_limits,
        #     'resume': (len(self._config.resume) > 0),
        #     'visualizer': self._config.visualizer,
        #     'time_attr': 'epoch',
        #     'reward_attr': 'map_reward',
        #     'dist_ip_addrs': self._config.dist_ip_addrs,
        #     'searcher': self._config.search_strategy,
        #     'search_options': self._config.search_options,
        # }
        # if self._config.search_strategy == 'hyperband':
        #     self._config.scheduler_options.update({
        #         'searcher': 'random',
        #         'max_t': self._config.epochs,
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
            search_options=self._config.search_options,
            nthreads_per_trial=self._config.nthreads_per_trial,
            ngpus_per_trial=self._config.ngpus_per_trial,
            checkpoint=self._config.checkpoint,
            num_trials=self._config.num_trials,
            time_out=self._config.time_limits,
            resume=(len(self._config.resume) > 0),
            visualizer=self._config.visualizer,
            time_attr='epoch',
            reward_attr='map_reward',
            dist_ip_addrs=self._config.dist_ip_addrs,
            epochs=self._config.epochs)

    def fit(self):
        results = self.run_fit(_train_object_detection, self._config.search_strategy,
                               self._config.scheduler_options)
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        best_config = sample_config(_train_object_detection.args, results['best_config'])
        self._logger.info('The best config: {}'.format(best_config))

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('BEST CONFIG:', best_config)
        print('BEST CONFIG TYPE:', type(best_config))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        estimator = self._estimator(best_config)
        estimator.put_parameters(results.pop('model_params'))
        return estimator
