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
from gluoncv.auto.data.dataset import ImageClassificationDataset

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassificationDataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

def test_image_classification_estimator():
    from gluoncv.auto.estimators import TorchImageClassificationEstimator
    # est = TorchImageClassificationEstimator({'train': {'epochs': 1}, 'misc': {'log_interval': 5}, 'gpus': ()})
    est = TorchImageClassificationEstimator({'train': {'epochs': 1}, 'misc': {'log_interval': 5}})
    res = est.fit(IMAGE_CLASS_DATASET)


if __name__ == '__main__':
    import nose
    nose.runmodule()
