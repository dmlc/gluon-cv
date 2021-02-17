from gluoncv.auto.tasks import ImageClassification
from gluoncv.auto.tasks import ObjectDetection
import autogluon.core as ag
import time

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
OBJECT_DETCTION_DATASET = ObjectDetection.Dataset.from_voc('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
OBJECT_DETECTION_TRAIN, OBJECT_DETECTION_VAL, OBJECT_DETECTION_TEST = OBJECT_DETCTION_DATASET.random_split(val_size=0.3, test_size=0.2)

def test_image_classification():
    from gluoncv.auto.tasks import ImageClassification
    task = ImageClassification({'num_trials': 1, 'epochs': 1})
    classifier = task.fit(IMAGE_CLASS_DATASET)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(IMAGE_CLASS_TEST)

def test_image_classification_custom_net():
    from gluoncv.auto.tasks import ImageClassification
    from gluoncv.model_zoo import get_model
    net = get_model('resnet18_v1')
    task = ImageClassification({'num_trials': 1, 'epochs': 1, 'custom_net': net})
    classifier = task.fit(IMAGE_CLASS_DATASET)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(IMAGE_CLASS_TEST)

def test_object_detection_estimator():
    from gluoncv.auto.tasks import ObjectDetection
    task = ObjectDetection({'num_trials': 1, 'epochs': 1})
    detector = task.fit(OBJECT_DETECTION_TRAIN)
    assert task.fit_summary().get('valid_map', 0) > 0
    test_result = detector.predict(OBJECT_DETECTION_TEST)

def test_object_detection_estimator_transfer():
    from gluoncv.auto.tasks import ObjectDetection
    task = ObjectDetection({'num_trials': 1, 'epochs': 1, 'transfer': ag.Categorical('yolo3_darknet53_coco', 'ssd_512_resnet50_v1_voc'), 'estimator': 'ssd'})
    detector = task.fit(OBJECT_DETECTION_TRAIN)
    assert task.fit_summary().get('valid_map', 0) > 0
    test_result = detector.predict(OBJECT_DETECTION_TEST)

def test_time_out_image_classification():
    time_limit = 30
    from gluoncv.auto.tasks import ImageClassification
    task = ImageClassification({'num_trials': 1, 'epochs': 50})

    tic = time.time()
    classifier = task.fit(IMAGE_CLASS_DATASET, time_limit=time_limit)
    # check time_limit with a little bit overhead
    assert (time.time() - tic) < time_limit + 180

def test_time_out_detection():
    time_limit = 30
    from gluoncv.auto.tasks import ObjectDetection
    task = ObjectDetection({'num_trials': 1, 'epochs': 50, 'time_limits': time_limit})
    tic = time.time()
    detector = task.fit(OBJECT_DETECTION_TRAIN)
    # check time_limit with a little bit overhead
    assert (time.time() - tic) < time_limit + 180
