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
from nose.tools import nottest
from gluoncv.auto.estimators import TorchImageClassificationEstimator
from gluoncv.auto.tasks import ImageClassification, ImagePrediction
import autogluon.core as ag
from autogluon.core.scheduler.resource import get_cpu_count, get_gpu_count

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
IMAGE_REGRESS_DATASET, _, IMAGE_REGRESS_TEST = ImagePrediction.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip', no_class=True)

def test_image_regression_estimator():
    est = TorchImageClassificationEstimator({'img_cls': {'model': 'resnet18'}, 'train': {'epochs': 1}, 'gpus': list(range(get_gpu_count()))},
                                      problem_type='regression')
    res = est.fit(IMAGE_REGRESS_DATASET)
    assert res.get('valid_score', 3) < 3
    est.predict(IMAGE_REGRESS_TEST)
    est.predict(IMAGE_REGRESS_TEST.iloc[0]['image'])
    est.evaluate(IMAGE_REGRESS_TEST)
    est.predict_feature(IMAGE_REGRESS_TEST)
    est.predict_feature(IMAGE_REGRESS_TEST.iloc[0]['image'])
    # test save/load
    _save_load_test(est, 'img_regression.pkl')

def test_image_classification_estimator():
    est = TorchImageClassificationEstimator({'img_cls': {'model': 'resnet18'}, 'train': {'epochs': 1}, 'gpus': list(range(get_gpu_count()))})
    res = est.fit(IMAGE_CLASS_DATASET)
    est.predict(IMAGE_CLASS_TEST)
    est.predict(IMAGE_CLASS_TEST.iloc[0]['image'])
    est.evaluate(IMAGE_CLASS_DATASET)
    est.predict_feature(IMAGE_CLASS_TEST)
    est.predict_feature(IMAGE_CLASS_TEST.iloc[0]['image'])
    _save_load_test(est, 'test.pkl')

def test_image_classification_estimator_custom_net():
    import timm
    import torch.nn as nn
    m = timm.create_model('resnet18')
    m.fc = nn.Linear(512, 4)
    est = TorchImageClassificationEstimator({'img_cls': {'model': 'resnet18'}, 'train': {'epochs': 1}, 'gpus': list(range(get_gpu_count()))},
                                            net=m)
    res = est.fit(IMAGE_CLASS_DATASET)
    est.predict(IMAGE_CLASS_TEST)
    est.predict(IMAGE_CLASS_TEST.iloc[0]['image'])
    est.evaluate(IMAGE_CLASS_DATASET)
    est.predict_feature(IMAGE_CLASS_TEST)
    est.predict_feature(IMAGE_CLASS_TEST.iloc[0]['image'])
    _save_load_test(est, 'test.pkl')

def test_image_classification_estimator_cpu():
    est = TorchImageClassificationEstimator({'img_cls': {'model': 'resnet18'}, 'train': {'epochs': 1}, 'gpus': ()})
    res = est.fit(IMAGE_CLASS_DATASET)
    est.predict(IMAGE_CLASS_TEST)
    est.predict(IMAGE_CLASS_TEST.iloc[0]['image'])
    est.evaluate(IMAGE_CLASS_DATASET)
    est.predict_feature(IMAGE_CLASS_TEST)
    est.predict_feature(IMAGE_CLASS_TEST.iloc[0]['image'])
    _save_load_test(est, 'test.pkl')

@nottest
def test_config_combination():
    for _ in range(100):
        test_config = build_config().rand
        est = TorchImageClassificationEstimator(test_config)
        res = est.fit(IMAGE_CLASS_DATASET)
        est.predict(IMAGE_CLASS_TEST)
        est.predict_feature(IMAGE_CLASS_TEST)
        _save_load_test(est, 'test.pkl')

def _save_load_test(est, filename):
    est._cfg.unfreeze()
    est._cfg.gpus = list(range(16))  # invalid cfg, check if load can restore succesfully
    est.save(filename)
    est2 = est.__class__.load(filename)
    return est2

@ag.func(
    pretrained=ag.space.Categorical(True, False),
    global_pool_type=ag.space.Categorical('fast', 'avg', 'max', 'avgmax'),
    sync_bn=ag.space.Categorical(True, False),
    no_aug=ag.space.Categorical(True, False),
    mixup=ag.space.Categorical(0.0, 0.5),
    cutmix=ag.space.Categorical(0.0, 0.5),
    model_ema=ag.space.Categorical(True, False),
    model_ema_force_cpu=ag.space.Categorical(True, False),
    save_images=ag.space.Categorical(True, False),
    pin_mem=ag.space.Categorical(True, False),
    use_multi_epochs_loader=ag.space.Categorical(True, False),
    amp=ag.space.Categorical(True, False),
    apex_amp=ag.space.Categorical(True, False),
    native_amp=ag.space.Categorical(True, False),
    prefetcher=ag.space.Categorical(True, False),
    interpolation=ag.space.Categorical('random', 'bilinear', 'bicubic'),
    batch_size=ag.space.Categorical(1,2,4,8,16,32),
    hflip=ag.space.Categorical(0.0, 0.5, 1.0),
    vflip=ag.space.Categorical(0.0, 0.5, 1.0),
    train_interpolation=ag.space.Categorical('random', 'bilinear', 'bicubic'),
    num_workers=ag.space.Categorical(1,2,4,8),
    tta=ag.space.Categorical(0,1)
)
def build_config(pretrained, global_pool_type, sync_bn, no_aug, mixup, cutmix, model_ema,
                model_ema_force_cpu, save_images, pin_mem, use_multi_epochs_loader,
                 amp, apex_amp, native_amp, prefetcher, interpolation, batch_size, hflip, vflip,
                 train_interpolation, num_workers, tta):
    config = {
        'model': {
            'model': 'resnet50',
            'pretrained': pretrained,
            'global_pool_type': global_pool_type,
        },
        'dataset': {
            'interpolation': interpolation
        },
        'train': {
            'batch_size': batch_size,
            'sync_bn': sync_bn,
        },
        'augmentation': {
            'no_aug': no_aug,
            'mixup': mixup,
            'cutmix': cutmix,
            'hflip': hflip,
            'vflip': vflip,
            'train_interpolation': train_interpolation,
        },
        'model_ema': {
            'model_ema': model_ema,
            'model_ema_force_cpu': model_ema_force_cpu,
        },
        'misc': {
            'num_workers': num_workers,
            'save_images': save_images,
            'pin_mem': pin_mem,
            'tta': tta,
            'use_multi_epochs_loader': use_multi_epochs_loader,
            'amp': amp,
            'apex_amp': apex_amp,
            'native_amp': native_amp,
            'prefetcher': prefetcher,
        }
    }
    if config['augmentation']['mixup'] or config['augmentation']['cutmix']:
        config['train']['batch_size'] = 2
    config['train']['epochs'] = 1
    config['gpus'] = list(range(get_gpu_count()))
    config['misc']['apex_amp'] = False # apex amp cause mem leak: https://github.com/NVIDIA/apex/issues/439
    return config

if __name__ == '__main__':
    import nose
    nose.runmodule()
