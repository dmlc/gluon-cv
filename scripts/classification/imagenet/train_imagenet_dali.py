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

import argparse
import logging
logging.basicConfig(level=logging.INFO)
from common import data, dali, fit
import mxnet as mx
import numpy as np
from gluoncv.model_zoo import get_model
import horovod.mxnet as hvd

def set_imagenet_aug(aug):
    # standard data augmentation setting for imagenet training
    aug.set_defaults(rgb_mean='123.68,116.779,103.939', rgb_std='58.393,57.12,57.375')
    aug.set_defaults(random_crop=0, random_resized_crop=1, random_mirror=1)
    aug.set_defaults(min_random_area=0.08)
    aug.set_defaults(max_random_aspect_ratio=4./3., min_random_aspect_ratio=3./4.)
    aug.set_defaults(brightness=0.4, contrast=0.4, saturation=0.4, pca_noise=0.1)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet-1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    dali.add_dali_args(parser)
    data.add_data_aug_args(parser)
    parser.set_defaults(
        # network
        network          = 'resnet',
        num_layers       = 50,

        # data
        resize           = 256,
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 90,
        lr_step_epochs   = '30,60,80',
        dtype            = 'float32'
    )
    args = parser.parse_args()
    gpus = list(map(int, filter(None, args.gpus.split(','))))
    mx.util.is_enough_gpus(gpus)

    if args.use_imagenet_data_augmentation:
        set_imagenet_aug(parser)

    if not args.use_dali:
        data.set_data_aug_level(parser, 0)

    # load network
    from importlib import import_module
    if args.network == "resnet-v1b" and args.num_layers == 50:
        net = get_model('resnet50_v1b', ctx=[mx.gpu(i) for i in gpus],
                        pretrained=False, classes=args.num_classes, last_gamma=args.bn_gamma_init0)
        d = mx.sym.var('data')
        if args.dtype == 'float16':
            d = mx.sym.Cast(data=d, dtype=np.float16)
        net.cast(args.dtype)
        out = net(d)
        if args.dtype == 'float16':
            out = mx.sym.Cast(data=out, dtype=np.float32)
        sym = mx.sym.SoftmaxOutput(out, name='softmax')
    else:
        net = import_module('symbols.'+args.network)
        sym = net.get_symbol(**vars(args))

    # Horovod: initialize Horovod
    if 'horovod' in args.kv_store:
        hvd.init()

    # train
    if args.use_dali:
        if args.benchmark != 0:
            raise ValueError("dali cannot be used with benchmark " +\
                             "(You passed: --use-dali --benchmark {})".format(args.benchmark))
        fit.fit(args, sym, dali.get_rec_iter)
    else:
        fit.fit(args, sym, data.get_rec_iter)
