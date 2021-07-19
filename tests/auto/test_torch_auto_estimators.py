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
from gluoncv.auto.estimators import TorchImageClassificationEstimator
from gluoncv.auto.data.dataset import ImageClassificationDataset
from autogluon.core.scheduler.resource import get_cpu_count, get_gpu_count

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassificationDataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

def test_image_classification_estimator():
    est = TorchImageClassificationEstimator({'model': {'model': 'resnet50'}, 'train': {'epochs': 100}, 'misc': {'log_interval': 5}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(IMAGE_CLASS_DATASET)
    est.predict(IMAGE_CLASS_TEST)
    est.predict_feature(IMAGE_CLASS_TEST)
    _save_load_test(est, 'test.pkl')

# def test_image_classification_estimator_cpu():
#     est = TorchImageClassificationEstimator({'model': {'model': 'resnet50'}, 'train': {'epochs': 1}, 'misc': {'log_interval': 5}, 'gpus': ()})
#     res = est.fit(IMAGE_CLASS_DATASET)
#     est.predict(IMAGE_CLASS_TEST)
#     est.predict_feature(IMAGE_CLASS_TEST)
#     _save_load_test(est, 'test.pkl')

def _save_load_test(est, filename):
    est._cfg.unfreeze()
    est._cfg.gpus = list(range(16))  # invalid cfg, check if load can restore succesfully
    est.save(filename)
    est2 = est.__class__.load(filename)
    return est2

if __name__ == '__main__':
    import nose
    nose.runmodule()
