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

"""utility functions providing the interface between dali & training"""

import warnings
import sys
import numpy as np
import horovod.mxnet as hvd
from mxnet.io import DataBatch, DataIter
import mxnet as mx
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator


_Reader = "Reader"      # default reader name
_mean_pixel = [255 * x for x in (0.485, 0.456, 0.406)]
_std_pixel = [255 * x for x in (0.229, 0.224, 0.225)]

synonyms_list = [('--separ-val', ('--dali-separ-val'))]

def add_dali_pipeline_args(parser, task_args=None, do_parsing=True):
    """Adding dali specific arguments for pipeline.
    Parameters
    ----------
    parser : argparse.ArgumentParser
    task_args : list ot tuple, which elements define additional parameters
    do_parsing : when True, the parsing is executed
    """
    def are_synonyms(nameA, nameB):
        if nameA == nameB:
            return True
        for synonims in synonyms_list:
            if synonims[0] == nameA:
                return nameB in synonims[1]
        return False

    default_args = [('--separ-val', 'store_true', 'each process will perform independent validation on whole val-set'),
                    ('--dali-threads', int, 3, 'number of threads per GPU for DALI'),
                    ('--dali-prefetch-queue', int, 3, 'DALI prefetch queue depth'),
                    ('--dali-nvjpeg-memory-padding', int, 16, 'Memory padding value for nvJPEG (in MB)'),
                    ('--num_examples', int, -1, 'Number of training examples to be used, "-1" - the full training set'),
                    ('--reader_name', str, "", 'Reader name')]

    # Check for duplicate arguments
    for arg in task_args:
        for def_arg in default_args:
            if are_synonyms(def_arg[0], arg[0]):
                default_args.remove(def_arg)  # Removing default argument
                break

    group = parser.add_argument_group('DALI', 'pipeline and augumentation')
    for args in [task_args, default_args]:
        for arg in args:
            if len(arg) == 3:
                group.add_argument(arg[0], action=arg[1], help=arg[2])
            else:
                group.add_argument(arg[0], type=arg[1], default=arg[2], help=arg[3])

    return parser.parse_args() if do_parsing else parser

def get_attribute(args, attr_name, def_value=None):
    attr = def_value
    if hasattr(args, attr_name):
        attr = hasattr(args, attr_name)
        if isinstance(attr, str):
            attr = list(map(float, attr.split(',')))
    return attr


class ImageReader:
    """
    Class providing image reading
    """
    def __init__(self, rec_path, idx_path, shard_id, num_shards, random_shuffle, reader_name=_Reader):
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=random_shuffle, shard_id=shard_id, num_shards=num_shards)
        self.reader_name = reader_name if isinstance(reader_name, str) and reader_name != "" else _Reader

class ImageMethods:
    """
    Class providing image related methods: decoding, resize, crop
    """
    def __init__(self, nvjpeg_padding, crop_shape, output_layout, pad_output, dtype,
                 random_crop=None, random_resize=False, args=None, dali_cpu=False):
        if dali_cpu:
            dali_device = "cpu"
            decoder_device = "cpu"
        else:
            dali_device = "gpu"
            decoder_device = "mixed"

        if random_crop:
            if args is None:
                type_crop = type(random_crop)
                len = len(random_crop) if type_crop is tuple or type_crop is list else 0
                self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                         device_memory_padding=nvjpeg_padding,
                                                         host_memory_padding=nvjpeg_padding,
                                                         random_aspect_ratio=random_crop[0] if len > 0 else [0.75,1.33],
                                                         random_area=random_crop[1] if len > 1 else [0.08,1.0],
                                                         num_attempts=random_crop[2] if len > 2 else 10)
            else:
                self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                         device_memory_padding=nvjpeg_padding,
                                                         host_memory_padding=nvjpeg_padding)
        else:
            self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
                                           device_memory_padding=nvjpeg_padding,
                                           host_memory_padding=nvjpeg_padding)
        if random_resize:
            self.resize = ops.RandomResizedCrop(device=dali_device, size=crop_shape)
        else:
            self.resize = ops.Resize(device=dali_device, resize_x=crop_shape[0], resize_y=crop_shape[1])

        rgb_mean = get_attribute(args, 'rgb_mean', _mean_pixel)
        rgb_std = get_attribute(args, 'rgb_std', _std_pixel)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
                                            mean=rgb_mean, std=rgb_std)

class HybridTrainPipe(Pipeline, ImageReader, ImageMethods):
    """
    Pypeline for hybridized training.
    """
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', args=None, dali_cpu=False,
                 reader_name=_Reader, random_crop=False, random_resize=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12+device_id,
                                              prefetch_queue_depth=prefetch_queue)
        ImageReader.__init__(self, rec_path, idx_path, shard_id, num_shards, True, reader_name=reader_name)
        ImageMethods.__init__(self, nvjpeg_padding, crop_shape, output_layout, pad_output, dtype,
                              random_crop=random_crop, random_resize=random_resize, args=args, dali_cpu=dali_cpu)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name=self.reader_name)
        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline, ImageReader, ImageMethods):
    """
    Pypeline for hybridized evaluation.
    """
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', random_resize=False,
                 args=None, dali_cpu=False, reader_name=_Reader):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12+device_id,
                                            prefetch_queue_depth=prefetch_queue)
        ImageReader.__init__(self, rec_path, idx_path, shard_id, num_shards, False, reader_name=reader_name)
        ImageMethods.__init__(self, nvjpeg_padding, crop_shape, output_layout, pad_output, dtype, args=args,
                              random_resize=random_resize, dali_cpu=dali_cpu)
        self.resize = ops.Resize(device="gpu", resize_shorter=resize_shp) if resize_shp else None

    def define_graph(self):
        self.jpegs, self.labels = self.input(name=self.reader_name)
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images)
        return [output, self.labels]

class SyntheticDataIter(DataIter):
    """
    Iterator for synthetic data.
    """
    def __init__(self, num_classes, data_shape, epoch_size, dtype, gpus, layout):
        super(SyntheticDataIter, self).__init__()
        self.batch_size = data_shape[0]
        self.cur_sample = 0
        self.epoch_size = epoch_size
        self.dtype = dtype
        self.gpus = gpus
        self._num_gpus = len(gpus)
        self.data_shape = data_shape
        self.layout = layout
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = [mx.nd.array(data, dtype=self.dtype, ctx=mx.gpu(i)) for i in gpus]
        self.label = [mx.nd.array(label, dtype=np.float32, ctx=mx.Context('cpu_pinned', 0)) for _ in gpus]

    def __iter__(self):
        return self

    @property
    def provide_data(self):
        data_shape = (self.data_shape[0] * self._num_gpus,) + self.data_shape[1:]
        return [mx.io.DataDesc('data', data_shape, self.dtype, self.layout)]

    @property
    def provide_label(self):
        label_shape = (self.batch_size * self._num_gpus,)
        return [mx.io.DataDesc('softmax_label', label_shape, np.float32)]

    def next(self):
        if self.cur_sample <= self.epoch_size:
            self.cur_sample += self.batch_size * self._num_gpus
            return [DataBatch(data=(d,),
                              label=(l,),
                              pad=0)
                    for d, l in zip(self.data, self.label)]
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_sample = 0


def get_rec_pipeline_iter(args, kv=None, dali_cpu=None, random_crop=True, prop_args=False):
    """Constructing pipeline iterators.
    Parameters
    ----------
    parser : argparse.ArgumentParser
    kv : list ot tuple, which elements define additional parameters
    random_crop : tuple, list OR boolean
                  when tuple OR list: if presented, the first, second and third elements define
                    - aspect ratio
                    - area
                    - number of attempts.
                  when True OR the some elements of tuple or list are not there, the default values will be used for
                    - aspect ratio: [0.75, 1.33],
                    - area: [0.08, 1.0],
                    - number of attempts 10
                  when False, only randomly resized crop will be used
    """
    # target shape is final shape of images pipelined to network;
    # all images will be cropped to this size
    target_shape = args.image_shape
    if isinstance(target_shape, str):
        target_shape = tuple([int(l) for l in target_shape.split(',')]) # filter to not encount eventually empty strings

    if hasattr(args, "gpus"):
        gpus = args.gpus
        if isinstance(gpus, str):
            gpus = list(map(int, filter(None, gpus.split(',')))) # filter to not encount eventually empty strings
    else:
        gpus = [hvd.local_rank()] if 'horovod' in args.kvstore else range(args.n_GPUs)

    batch_size = args.batch_size
    if 'synthetic' in args and args.synthetic == 1:
        print("Using synthetic data", file=sys.stderr)
        target_shape = target_shape[1:]+target_shape[:1] if args.input_layout == 'NHWC' else target_shape
        data_shape = (batch_size,) + target_shape
        train = SyntheticDataIter(1000, data_shape, args.num_examples, args.dtype, gpus, args.input_layout)
        return (train, None)

    print("Using DALI", file=sys.stderr)
    pad_output = target_shape[0] == 4
    num_threads = args.dali_threads
    num_validation_threads = args.dali_validation_threads if hasattr(args, "dali_validation_threads") else num_threads

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC if args.input_layout == 'NHWC' else types.NCHW

    if 'horovod' in (args.kvstore if hasattr(args, "kvstore") else args.kv_store):
        rank = hvd.rank()
        nWrk = hvd.size()
    else:
        rank = kv.rank if kv else 0
        nWrk = kv.num_workers if kv else 1

    num_shards = len(gpus)*nWrk

    reade_flg = hasattr(args, "reader_name") and isinstance(args.reader_name, str) and args.reader_name != ""
    reader_name = args.reader_name if reade_flg else _Reader

    rec_path = args.data_train if hasattr(args, "data_train") else args.data_dir + "/train.rec"
    idx_path = args.data_train_idx if hasattr(args, "data_train_idx") else args.data_dir + "/train.idx"

    if not dali_cpu:
        batch_size = args.batch_size // nWrk // len(gpus)
    nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024

    trainpipes = [HybridTrainPipe(batch_size=batch_size,
                                  num_threads=num_threads,
                                  device_id=gpu_id,
                                  rec_path=rec_path,
                                  idx_path=idx_path,
                                  shard_id=gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards=num_shards,
                                  crop_shape=target_shape[1:],
                                  output_layout=output_layout,
                                  pad_output=pad_output,
                                  dtype=args.dtype,
                                  nvjpeg_padding=nvjpeg_padding,
                                  prefetch_queue=args.dali_prefetch_queue,
                                  dali_cpu=dali_cpu,
                                  random_crop=random_crop,
                                  random_resize=not random_crop,
                                  args=args if prop_args else None,
                                  reader_name=reader_name) for gpu_id in gpus]

    val_flg = (hasattr(args, "data_val") and args.data_val is not None) and not (hasattr(args, "no_val") and args.no_val)
    if val_flg:
        rec_path = args.data_val if hasattr(args, "data_val") else args.data_dir + "/val.rec"
        idx_path = args.data_val_idx if hasattr(args, "data_val_idx") else args.data_dir + "/val.idx"
        separ_val_flag = hasattr(args, "dali_separ_val") or hasattr(args, "separ_val")

        # resize is default base length of shorter edge for dataset;
        # all images will be reshaped to this size
        resize_shp = int(args.resize) if hasattr(args, "resize") else args.data_val_resize

        valpipes = [HybridValPipe(batch_size=batch_size,
                                  num_threads=num_validation_threads,
                                  device_id=gpu_id,
                                  rec_path=rec_path,
                                  idx_path=idx_path,
                                  shard_id=0 if separ_val_flag else gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards=1 if separ_val_flag else num_shards,
                                  crop_shape=target_shape[1:],
                                  resize_shp=resize_shp,
                                  output_layout=output_layout,
                                  pad_output=pad_output,
                                  dtype=args.dtype,
                                  nvjpeg_padding=nvjpeg_padding,
                                  prefetch_queue=args.dali_prefetch_queue,
                                  dali_cpu=dali_cpu,
                                  random_resize=dali_cpu is None,
                                  args=args if prop_args else None,
                                  reader_name=reader_name) for gpu_id in gpus]
    trainpipes[0].build()
    if val_flg:
        valpipes[0].build()

    if args.num_examples < trainpipes[0].epoch_size(reader_name):
        warnings.warn("{} training examples will be used, although full training set contains {} examples".
                      format(args.num_examples, trainpipes[0].epoch_size(reader_name)))

    train_size = val_size = -1
    reader = args.reader_name if reade_flg else None
    if reader is None:
        train_size = args.num_examples // nWrk
        if val_flg:
            val_size = valpipes[0].epoch_size(reader_name)
            if not separ_val_flag:
                val_size = val_size // nWrk
                if rank < valpipes[0].epoch_size(reader_name):
                    val_size += 1

    dali_train_iter = DALIClassificationIterator(trainpipes, train_size, reader)
    dali_val_iter = DALIClassificationIterator(valpipes, val_size, reader, fill_last_batch=False) if val_flg else None
    return dali_train_iter, dali_val_iter
