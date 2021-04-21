# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test auto estimators"""
from gluoncv.auto.tasks import ImageClassification
from gluoncv.auto.tasks import ObjectDetection
from autogluon.core.scheduler.resource import get_cpu_count, get_gpu_count

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
OBJECT_DETCTION_DATASET = ObjectDetection.Dataset.from_voc('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
OBJECT_DETECTION_TRAIN, OBJECT_DETECTION_VAL, OBJECT_DETECTION_TEST = OBJECT_DETCTION_DATASET.random_split(val_size=0.1, test_size=0.1)

def test_image_classification_estimator():
    from gluoncv.auto.estimators import ImageClassificationEstimator
    est = ImageClassificationEstimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(IMAGE_CLASS_DATASET)
    assert res.get('valid_acc', 0) > 0
    test_result = est.predict(IMAGE_CLASS_TEST)
    test_result = est.predict(IMAGE_CLASS_TEST, with_proba=True)
    est.predict(IMAGE_CLASS_TEST.iloc[0]['image'])
    evaluate_result = est.evaluate(IMAGE_CLASS_TEST)
    feature = est.predict_feature(IMAGE_CLASS_TEST)
    est.predict_feature(IMAGE_CLASS_TEST.iloc[0]['image'])
    # test save/load
    _save_load_test(est, 'imgcls.pkl')

def test_image_classification_estimator_custom_net_optimizer():
    from gluoncv.auto.estimators import ImageClassificationEstimator
    from gluoncv.model_zoo import get_model
    from mxnet.optimizer import Adam
    net = get_model('resnet18_v1')
    optim = Adam(learning_rate=0.01, wd=1e-3)
    est = ImageClassificationEstimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))},
        net=net, optimizer=optim)
    res = est.fit(IMAGE_CLASS_DATASET)
    assert res.get('valid_acc', 0) > 0
    feat = est.predict_feature(IMAGE_CLASS_TEST)
    est.save('test_image_classification.pkl')
    est = ImageClassificationEstimator.load('test_image_classification.pkl')
    test_result = est.predict(IMAGE_CLASS_TEST)
    evaluate_result = est.evaluate(IMAGE_CLASS_TEST)
    feature = est.predict_feature(IMAGE_CLASS_TEST)

def test_center_net_estimator():
    from gluoncv.auto.estimators import CenterNetEstimator
    est = CenterNetEstimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(OBJECT_DETECTION_TRAIN)
    assert res.get('valid_map', 0) > 0
    test_result = est.predict(OBJECT_DETECTION_TEST)
    est.predict(OBJECT_DETECTION_TEST.iloc[0]['image'])
    evaluate_result = est.evaluate(OBJECT_DETECTION_VAL)
    # test save/load
    _save_load_test(est, 'center_net.pkl')

def test_ssd_estimator():
    from gluoncv.auto.estimators import SSDEstimator
    est = SSDEstimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(OBJECT_DETECTION_TRAIN)
    assert res.get('valid_map', 0) > 0
    test_result = est.predict(OBJECT_DETECTION_TEST)
    est.predict(OBJECT_DETECTION_TEST.iloc[0]['image'])
    evaluate_result = est.evaluate(OBJECT_DETECTION_VAL)
    # test save/load
    _save_load_test(est, 'ssd.pkl')

def test_yolo3_estimator():
    from gluoncv.auto.estimators import YOLOv3Estimator
    est = YOLOv3Estimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(OBJECT_DETECTION_TRAIN)
    assert res.get('valid_map', 0) > 0
    test_result = est.predict(OBJECT_DETECTION_TEST)
    est.predict(OBJECT_DETECTION_TEST.iloc[0]['image'])
    evaluate_result = est.evaluate(OBJECT_DETECTION_VAL)
    # test save/load
    _save_load_test(est, 'yolo3.pkl')

def test_frcnn_estimator():
    from gluoncv.auto.estimators import FasterRCNNEstimator
    est = FasterRCNNEstimator({'train': {'epochs': 1}, 'gpus': list(range(get_gpu_count()))})
    OBJECT_DETECTION_TRAIN_MINI, OBJECT_DETECTION_VAL_MINI, OBJECT_DETECTION_TEST_MINI = OBJECT_DETECTION_TRAIN.random_split(
        val_size=0.3, test_size=0.2)
    res = est.fit(OBJECT_DETECTION_TRAIN_MINI)
    assert res.get('valid_map', 0) > 0
    test_result = est.predict(OBJECT_DETECTION_TEST_MINI)
    est.predict(OBJECT_DETECTION_TEST.iloc[0]['image'])
    evaluate_result = est.evaluate(OBJECT_DETECTION_VAL_MINI)
    # test save/load
    _save_load_test(est, 'frcnn.pkl')

def _save_load_test(est, filename):
    est._cfg.unfreeze()
    est._cfg.gpus = list(range(16))  # invalid cfg, check if load can restore succesfully
    est.save(filename)
    est2 = est.__class__.load(filename)

if __name__ == '__main__':
    import nose
    nose.runmodule()
