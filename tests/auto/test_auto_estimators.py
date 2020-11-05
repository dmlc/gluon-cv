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

def test_image_classification_estimator():
    from gluoncv.auto.estimators import ImageClassificationEstimator
    est = ImageClassificationEstimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(IMAGE_CLASS_DATASET)
    assert res.get('valid_acc', 0) > 0
    test_result = est.predict(IMAGE_CLASS_TEST)
    evaluate_result = est.evaluate(IMAGE_CLASS_TEST)

def test_center_net_estimator():
    from gluoncv.auto.estimators import CenterNetEstimator
    est = CenterNetEstimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(OBJECT_DETCTION_DATASET)
    assert res.get('valid_map', 0) > 0
    _, _, test_data = OBJECT_DETCTION_DATASET.random_split()
    test_result = est.predict(test_data)
    evaluate_result = est.evaluate(test_data)

def test_ssd_estimator():
    from gluoncv.auto.estimators import SSDEstimator
    est = SSDEstimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(OBJECT_DETCTION_DATASET)
    assert res.get('valid_map', 0) > 0
    _, _, test_data = OBJECT_DETCTION_DATASET.random_split()
    test_result = est.predict(test_data)
    evaluate_result = est.evaluate(test_data)

def test_yolo3_estimator():
    from gluoncv.auto.estimators import YOLOv3Estimator
    est = YOLOv3Estimator({'train': {'epochs': 1, 'batch_size': 8}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(OBJECT_DETCTION_DATASET)
    assert res.get('valid_map', 0) > 0
    _, _, test_data = OBJECT_DETCTION_DATASET.random_split()
    test_result = est.predict(test_data)
    evaluate_result = est.evaluate(test_data)

def test_frcnn_estimator():
    from gluoncv.auto.estimators import FasterRCNNEstimator
    est = FasterRCNNEstimator({'train': {'epochs': 1}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(OBJECT_DETCTION_DATASET)
    assert res.get('valid_map', 0) > 0
    _, _, test_data = OBJECT_DETCTION_DATASET.random_split()
    test_result = est.predict(test_data)
    evaluate_result = est.evaluate(test_data)

if __name__ == '__main__':
    import nose
    nose.runmodule()
