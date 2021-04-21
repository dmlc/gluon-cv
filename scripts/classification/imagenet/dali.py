# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator, LastBatchPolicy
import horovod.mxnet as hvd


def add_dali_args(parser):
    group = parser.add_argument_group('DALI data backend', 'entire group applies only to dali data backend')
    group.add_argument('--dali-separ-val', action='store_true',
                      help='each process will perform independent validation on whole val-set')
    group.add_argument('--dali-threads', type=int, default=3, help="number of threads" +\
                       "per GPU for DALI")
    group.add_argument('--dali-validation-threads', type=int, default=10, help="number of threads" +\
                       "per GPU for DALI for validation")
    group.add_argument('--dali-prefetch-queue', type=int, default=2, help="DALI prefetch queue depth")
    group.add_argument('--dali-nvjpeg-memory-padding', type=int, default=64, help="Memory padding value for nvJPEG (in MB)")
    group.add_argument('--dali-fuse-decoder', type=int, default=1, help="0 or 1 whether to fuse decoder or not")
    group.add_argument('--flag', type=int, default=1, help="Flag")
    return parser

def add_data_args(parser):
    def int_list(x):
        return list(map(int, x.split(',')))

    data = parser.add_argument_group('Data')
    data.add_argument('--data-pred', type=str, help='the image on which run inference (only for pred mode)')
    data.add_argument('--image-shape', type=int_list, default=[3, 224, 224],
                      help='the image shape feed into the network')

    data.add_argument('--input-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the input data')
    data.add_argument('--conv-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the data assumed by the conv operation')
    data.add_argument('--batchnorm-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the data assumed by the batchnorm operation')
    data.add_argument('--pooling-layout', type=str, default='NCHW', choices=('NCHW', 'NHWC'),
                      help='the layout of the data assumed by the pooling operation')

    data.add_argument('--num-examples', type=int, default=1281167,
                      help="the number of training examples (doesn't work with mxnet data backend)")
    data.add_argument('--data-val-resize', type=int, default=256,
                      help='base length of shorter edge for validation dataset')
    data.add_argument('--kv-store', type=str, default='device', choices=('device', 'horovod'),
                      help='key-value store type')
    return data


def get_device_names(dali_cpu):
    return ("cpu", "cpu") if dali_cpu else ("gpu", "mixed")


class HybridTrainPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, prefetch_queue_depth = prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=True, shard_id=shard_id, num_shards=num_shards)

        dali_device, decoder_device = get_device_names(dali_cpu)
        if args.dali_fuse_decoder:
            self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                    device_memory_padding=nvjpeg_padding, host_memory_padding=nvjpeg_padding)
        else:
            self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
                                           device_memory_padding=nvjpeg_padding, host_memory_padding=nvjpeg_padding)

        if args.dali_fuse_decoder:
            self.resize = ops.Resize(device=dali_device, resize_x=crop_shape[1], resize_y=crop_shape[0])
        else:
            self.resize = ops.RandomResizedCrop(device=dali_device, size=crop_shape)

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
                                            mean=args.rgb_mean, std=args.rgb_std)
        self.coin = ops.random.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, prefetch_queue_depth=prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        dali_device, decoder_device = get_device_names(dali_cpu)
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
                                       device_memory_padding=nvjpeg_padding,
                                       host_memory_padding=nvjpeg_padding)
        self.resize = ops.Resize(device=dali_device, resize_shorter=resize_shp) if resize_shp else None
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
                                            mean=args.rgb_mean, std=args.rgb_std)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]


def get_rec_iter(args, kv=None, batch_fn=None, dali_cpu=False):
    devices = [0] if dali_cpu else args.gpus
    num_devices = len(devices)
    pad_output = (args.image_shape[0] == 4)

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC if args.input_layout == 'NHWC' else types.NCHW

    if 'horovod' in args.kv_store:
        rank = hvd.rank()
        nWrk = hvd.size()
    else:
        rank = kv.rank if kv else 0
        nWrk = kv.num_workers if kv else 1

    batch_size = args.batch_size // nWrk * num_devices

    trainpipes = [HybridTrainPipe(args           = args,
                                  batch_size     = batch_size,
                                  num_threads    = args.dali_threads,
                                  device_id      = dev_id,
                                  rec_path       = args.rec_train,
                                  idx_path       = args.rec_train_idx,
                                  shard_id       = devices.index(dev_id) + num_devices*rank,
                                  num_shards     = num_devices*nWrk,
                                  crop_shape     = args.image_shape[1:],
                                  output_layout  = output_layout,
                                  dtype          = args.dtype,
                                  pad_output     = pad_output,
                                  dali_cpu       = dali_cpu,
                                  nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                  prefetch_queue = args.dali_prefetch_queue) for dev_id in devices]
    trainpipes[0].build()
    num_examples = trainpipes[0].epoch_size("Reader")
    if args.num_examples < num_examples:
        warnings.warn("{} training examples will be used, although full training set contains {} examples".format(args.num_examples, num_examples))

    train_examples = args.num_examples // nWrk
    dali_train_iter = DALIClassificationIterator(trainpipes, train_examples)
    if not args.rec_val:
        return dali_train_iter, None, batch_fn

    valpipes = [HybridValPipe(args           = args,
                              batch_size     = batch_size,
                              num_threads    = args.dali_validation_threads,
                              device_id      = dev_id,
                              rec_path       = args.rec_val,
                              idx_path       = args.rec_val_idx,
                              shard_id       = 0 if args.dali_separ_val else devices.index(dev_id) + num_devices*rank,
                              num_shards     = 1 if args.dali_separ_val else num_devices*nWrk,
                              crop_shape     = args.image_shape[1:],
                              resize_shp     = args.data_val_resize,
                              output_layout  = output_layout,
                              dtype          = args.dtype,
                              pad_output     = pad_output,
                              dali_cpu       = dali_cpu,
                              nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                              prefetch_queue = args.dali_prefetch_queue) for dev_id in devices]
    valpipes[0].build()
    worker_val_examples = valpipes[0].epoch_size("Reader")
    if not args.dali_separ_val:
        adj = 1 if rank < worker_val_examples % nWrk else 0
        worker_val_examples = adj + worker_val_examples // nWrk

    dali_val_iter = DALIClassificationIterator(valpipes, worker_val_examples)
    return dali_train_iter, dali_val_iter, batch_fn
