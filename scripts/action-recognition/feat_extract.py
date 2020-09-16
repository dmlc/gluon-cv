import os
import sys
import time
import argparse
import logging
import math
import gc
import json

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord

def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from pre-trained models for video related tasks.')
    parser.add_argument('--data-dir', type=str, default='',
                        help='the root path to your data')
    parser.add_argument('--need-root', action='store_true',
                        help='if set to True, --data-dir needs to be provided as the root path to find your videos.')
    parser.add_argument('--data-list', type=str, default='',
                        help='the list of your data. You can either provide complete path or relative path.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='number of gpus to use. Use -1 for CPU')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                        help='enable using pretrained model from GluonCV.')
    parser.add_argument('--hashtag', type=str, default='',
                        help='hashtag for pretrained models.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--new-height', type=int, default=256,
                        help='new height of the resize image. default is 256')
    parser.add_argument('--new-width', type=int, default=340,
                        help='new width of the resize image. default is 340')
    parser.add_argument('--new-length', type=int, default=32,
                        help='new length of video sequence. default is 32')
    parser.add_argument('--new-step', type=int, default=1,
                        help='new step to skip video sequence. default is 1')
    parser.add_argument('--num-classes', type=int, default=400,
                        help='number of classes.')
    parser.add_argument('--ten-crop', action='store_true',
                        help='whether to use ten crop evaluation.')
    parser.add_argument('--three-crop', action='store_true',
                        help='whether to use three crop evaluation.')
    parser.add_argument('--video-loader', action='store_true', default=True,
                        help='if set to True, read videos directly instead of reading frames.')
    parser.add_argument('--use-decord', action='store_true', default=True,
                        help='if set to True, use Decord video loader to load data.')
    parser.add_argument('--slowfast', action='store_true',
                        help='if set to True, use data loader designed for SlowFast network.')
    parser.add_argument('--slow-temporal-stride', type=int, default=16,
                        help='the temporal stride for sparse sampling of video frames for slow branch in SlowFast network.')
    parser.add_argument('--fast-temporal-stride', type=int, default=2,
                        help='the temporal stride for sparse sampling of video frames for fast branch in SlowFast network.')
    parser.add_argument('--num-crop', type=int, default=1,
                        help='number of crops for each image. default is 1')
    parser.add_argument('--data-aug', type=str, default='v1',
                        help='different types of data augmentation pipelines. Supports v1, v2, v3 and v4.')
    parser.add_argument('--num-segments', type=int, default=1,
                        help='number of segments to evenly split the video.')
    parser.add_argument('--save-dir', type=str, default='./',
                        help='directory of saved results')
    opt = parser.parse_args()
    return opt

def read_data(opt, video_name, transform, video_utils):

    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_name, width=opt.new_width, height=opt.new_height)
    duration = len(decord_vr)

    opt.skip_length = opt.new_length * opt.new_step
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if opt.video_loader:
        if opt.slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
    else:
        raise RuntimeError('We only support video-based inference.')

    clip_input = transform(clip_input)

    if opt.slowfast:
        sparse_sampels = len(clip_input) // (opt.num_segments * opt.num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (opt.new_length, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if opt.new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

    return nd.array(clip_input)

def main(logger):
    opt = parse_args()
    logger.info(opt)
    gc.set_threshold(100, 5, 5)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # set env
    if opt.gpu_id == -1:
        context = mx.cpu()
    else:
        gpu_id = opt.gpu_id
        context = mx.gpu(gpu_id)

    # get data preprocess
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if opt.ten_crop:
        transform_test = transforms.Compose([
            video.VideoTenCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 10
    elif opt.three_crop:
        transform_test = transforms.Compose([
            video.VideoThreeCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 3
    else:
        transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=image_norm_mean, std=image_norm_std)
        opt.num_crop = 1

    # get model
    if opt.use_pretrained and len(opt.hashtag) > 0:
        opt.use_pretrained = opt.hashtag
    classes = opt.num_classes
    model_name = opt.model
    net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                    feat_ext=True, num_segments=opt.num_segments, num_crop=opt.num_crop)
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.resume_params != '' and not opt.use_pretrained:
        net.load_parameters(opt.resume_params, ctx=context)
        logger.info('Pre-trained model %s is successfully loaded.' % (opt.resume_params))
    else:
        logger.info('Pre-trained model is successfully loaded from the model zoo.')
    logger.info("Successfully built model {}".format(model_name))

    # get data
    anno_file = opt.data_list
    f = open(anno_file, 'r')
    data_list = f.readlines()
    logger.info('Load %d video samples.' % len(data_list))

    # build a pseudo dataset instance to use its children class methods
    video_utils = VideoClsCustom(root=opt.data_dir,
                                 setting=opt.data_list,
                                 num_segments=opt.num_segments,
                                 num_crop=opt.num_crop,
                                 new_length=opt.new_length,
                                 new_step=opt.new_step,
                                 new_width=opt.new_width,
                                 new_height=opt.new_height,
                                 video_loader=opt.video_loader,
                                 use_decord=opt.use_decord,
                                 slowfast=opt.slowfast,
                                 slow_temporal_stride=opt.slow_temporal_stride,
                                 fast_temporal_stride=opt.fast_temporal_stride,
                                 data_aug=opt.data_aug,
                                 lazy_init=True)

    start_time = time.time()
    for vid, vline in enumerate(data_list):
        video_path = vline.split()[0]
        video_name = video_path.split('/')[-1]
        if opt.need_root:
            video_path = os.path.join(opt.data_dir, video_path)
        video_data = read_data(opt, video_path, transform_test, video_utils)
        video_input = video_data.as_in_context(context)
        video_feat = net(video_input.astype(opt.dtype, copy=False))

        feat_file = '%s_%s_feat.npy' % (model_name, video_name)
        np.save(os.path.join(opt.save_dir, feat_file), video_feat.asnumpy())

        if vid > 0 and vid % opt.log_interval == 0:
            logger.info('%04d/%04d is done' % (vid, len(data_list)))

    end_time = time.time()
    logger.info('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))

if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    main(logger)
