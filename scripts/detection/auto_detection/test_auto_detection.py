import autogluon as ag

from gluoncv.auto.estimators.ssd import SSDEstimator
from gluoncv.auto.estimators.faster_rcnn import FasterRCNNEstimator
from gluoncv.auto.estimators.yolo import YoloEstimator
from gluoncv.auto.estimators.center_net import CenterNetEstimator
# from gluoncv.auto.estimators import SSDEstimator, FasterRCNNEstimator, YoloEstimator, CenterNetEstimator
from gluoncv.auto.tasks.object_detection import ObjectDetection


if __name__ == '__main__':
    args = {
        'dataset': 'voc',
        'meta_arch': 'yolo',
        'lr': ag.Categorical(1e-3, 5e-4),
        'epochs': 2,
        'num_trials': 2
    }

    if 'meta_arch' in args:
        if args['meta_arch'] == 'ssd':
            estimator = SSDEstimator
        elif args['meta_arch'] == 'faster_rcnn':
            estimator = FasterRCNNEstimator
        elif args['meta_arch'] == 'yolo3':
            estimator = YoloEstimator
        elif args['meta_arch'] == 'center_net':
            estimator = CenterNetEstimator
        else:
            raise NotImplementedError('%s is not implemented.', args['meta_arch'])
    else:
        estimator = None

    task = ObjectDetection(args, estimator)

    detector = task.fit()
    test_map = detector.evaluate()
    print("mAP on test dataset: {}".format(test_map[-1][-1]))
    print(test_map)
    # detector.save('final_model.model')
