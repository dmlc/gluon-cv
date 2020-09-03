import logging
import numpy as np

import autogluon as ag
from autogluon.core.decorator import sample_config
from autogluon.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.task import BaseTask
from autogluon.task.object_detection.dataset.voc import CustomVOCDetectionBase
from autogluon.utils import collect_params

from ... import data as gdata
from ... import utils as gutils
from ..estimators.base_estimator import ConfigDict, BaseEstimator
from ..estimators.ssd import SSDEstimator
from ..estimators.faster_rcnn import FasterRCNNEstimator
from ..estimators import YoloEstimator, CenterNetEstimator
from .auto_config import auto_args, config_to_nested


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

    try:
        estimator = args['estimator'](args, reporter=reporter)
        # training
        estimator.fit()
    except Exception as e:
        return str(e)

    # TODO: checkpointing needs to be done in a better way
    if args['final_fit']:
        return {'model_params': collect_params(estimator.net)}

    return {}


class ObjectDetection(BaseTask):
    def __init__(self, config, estimator=None, logger=None):
        super(ObjectDetection, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._estimator = estimator
        self._config = ConfigDict(config)

        if self._config.get('auto_search', True):
            # The strategies can be injected here, for example: automatic suggest some hyperparameters
            # based on the dataset statistics

            # get dataset statistics
            dataset_name = self._config.get('dataset', 'voc')
            dataset_root = self._config.get('dataset_root', '~/.mxnet/datasets/')
            if dataset_name == 'voc':
                train_dataset = gdata.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
            elif dataset_name == 'voc_tiny':
                train_dataset = CustomVOCDetectionBase(classes=('motorbike',),
                                                       root=dataset_root + 'tiny_motorbike',
                                                       splits=[('', 'trainval')])
            elif dataset_name == 'coco':
                train_dataset = gdata.COCODetection(splits=['instances_train2017'])
            else:
                # user needs to define a Dataset object "train_dataset" when using custom dataset
                train_dataset = self._config.get('train_dataset', None)

            # choose 100 examples to calculate average statistics
            num_examples = 100
            image_height_list = []
            image_width_list = []
            image_size_list = []
            num_objects_list = []
            object_height_list = []
            object_width_list = []
            object_size_list = []

            for i in range(num_examples):
                train_image, train_label = train_dataset[i]

                image_height_list.append(train_image.shape[0])
                image_width_list.append(train_image.shape[1])
                image_size_list.append(train_image.shape[0] * train_image.shape[1])

                bounding_boxes = train_label[:, :4]
                num_objects_list.append(bounding_boxes.shape[0])
                object_height_list.append(np.mean(bounding_boxes[:, 3] - bounding_boxes[:, 1]))
                object_width_list.append(np.mean(bounding_boxes[:, 2] - bounding_boxes[:, 0]))
                object_size_list.append(np.mean((bounding_boxes[:, 3] - bounding_boxes[:, 1]) *
                                                (bounding_boxes[:, 2] - bounding_boxes[:, 0])))

            num_images = len(train_dataset)
            image_height = np.mean(image_height_list)
            image_width = np.mean(image_width_list)
            image_size = np.mean(image_size_list)
            num_classes = len(train_dataset.CLASSES)
            num_objects = np.mean(num_objects_list)
            object_height = np.mean(object_height_list)
            object_width = np.mean(object_width_list)
            object_size = np.mean(object_size_list)

            print('number of training images:', num_images)
            print('average image height:', image_height)
            print('average image width:', image_width)
            print('average image size:', image_size)
            print('number of total object classes:', num_classes)
            print('average number of objects in an image:', num_objects)
            print('average bounding box height:', object_height)
            print('average bounding box width:', object_width)
            print('average bounding box size:', object_size)

            # specify 3 parts of config: dataset, model, training
            if num_images >= 1000:
                pass
            elif 100 <= num_images < 1000:
                pass
            else:
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

        # If only time_limits is given, the scheduler starts trials until the time limit is reached
        if self._config.num_trials is None and self._config.time_limits is None:
            self._config.num_trials = 2

        if estimator is None:
            estimator = [SSDEstimator, FasterRCNNEstimator, YoloEstimator, CenterNetEstimator]
        elif isinstance(estimator, (tuple, list)):
            pass
        else:
            assert issubclass(estimator, BaseEstimator)
            estimator = [estimator]

        config['estimator'] = ag.Categorical(*estimator)
        config['num_workers'] = nthreads_per_trial
        config['gpus'] = [int(i) for i in range(ngpus_per_trial)]
        config['seed'] = self._config.get('seed', 233)
        config['final_fit'] = False

        # automatically merge search configs according to user specified values
        args = auto_args(estimator, config)

        # _train_object_detection.register_args(**args)
        _train_object_detection.register_args(**config)

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

    def fit(self):
        results = self.run_fit(_train_object_detection, self._config.search_strategy,
                               self._config.scheduler_options)
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        best_config = sample_config(_train_object_detection.args, results['best_config'])
        # convert best config to nested form
        best_config = config_to_nested(best_config)
        self._logger.info('The best config: {}'.format(best_config))

        estimator = self._estimator(best_config)
        # TODO: checkpointing needs to be done in a better way
        estimator.put_parameters(results.pop('model_params'))
        return estimator
