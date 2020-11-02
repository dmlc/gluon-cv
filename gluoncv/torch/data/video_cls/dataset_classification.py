"""Customized dataloader for general video classification tasks."""
import os
import warnings
import numpy as np
from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset

from ..transforms.videotransforms import video_transforms, volume_transforms
from .multigrid_helper import multiGridSampler, MultiGridBatchSampler


__all__ = ['VideoClsDataset', 'build_dataloader', 'build_dataloader_test']


class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=False,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 use_multigrid=False):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.use_multigrid = use_multigrid and (mode == 'train')

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 2])

        if (mode == 'train'):
            if self.use_multigrid:
                self.MG_sampler = multiGridSampler()
                self.data_transform = []
                for alpha in range(self.MG_sampler.mod_long):
                    tmp = []
                    for beta in range(self.MG_sampler.mod_short):
                        info = self.MG_sampler.get_resize(alpha, beta)
                        scale_s = info[1]
                        tmp.append(video_transforms.Compose([
                            video_transforms.Resize(int(self.short_side_size / scale_s),
                                                    interpolation='bilinear'),
                            # TODO: multiscale corner cropping
                            video_transforms.RandomResize(ratio=(1, 1.25),
                                                          interpolation='bilinear'),
                            video_transforms.RandomCrop(size=(int(self.crop_size / scale_s),
                                                              int(self.crop_size / scale_s)))]))
                    self.data_transform.append(tmp)
            else:
                self.data_transform = video_transforms.Compose([
                    video_transforms.Resize(int(self.short_side_size),
                                            interpolation='bilinear'),
                    video_transforms.RandomResize(ratio=(1, 1.25),
                                                  interpolation='bilinear'),
                    video_transforms.RandomCrop(size=(int(self.crop_size),
                                                      int(self.crop_size)))])

            self.data_transform_after = video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.use_multigrid is True:
                index, alpha, beta = index
                info = self.MG_sampler.get_resize(alpha, beta)
                scale_t = info[0]
                data_transform_func = self.data_transform[alpha][beta]
            else:
                scale_t = 1
                data_transform_func = self.data_transform

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            buffer = data_transform_func(buffer)
            buffer = self.data_transform_after(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        # pylint: disable=line-too-long, bare-except, unnecessary-comprehension
        fname = self.data_path + sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def build_dataloader(cfg):
    """Build dataloader for training/validation"""
    train_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.TRAIN_ANNO_PATH,
                                    data_path=cfg.CONFIG.DATA.TRAIN_DATA_PATH,
                                    mode='train',
                                    use_multigrid=cfg.CONFIG.DATA.MULTIGRID,
                                    clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                    frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                    num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                                    num_crop=cfg.CONFIG.DATA.NUM_CROP,
                                    keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                    crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                    short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                    new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                    new_width=cfg.CONFIG.DATA.NEW_WIDTH)
    val_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.VAL_ANNO_PATH,
                                  data_path=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                  mode='validation',
                                  use_multigrid=cfg.CONFIG.DATA.MULTIGRID,
                                  clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                  frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                  num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                                  num_crop=cfg.CONFIG.DATA.NUM_CROP,
                                  keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                  crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                  short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                  new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                  new_width=cfg.CONFIG.DATA.NEW_WIDTH)

    if cfg.DDP_CONFIG.DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    mg_sampler = None
    if cfg.CONFIG.DATA.MULTIGRID:
        mg_sampler = MultiGridBatchSampler(train_sampler, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE,
                                           drop_last=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                                   num_workers=9, pin_memory=True,
                                                   batch_sampler=mg_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None),
            num_workers=9, sampler=train_sampler, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=9, sampler=val_sampler, pin_memory=True)

    return train_loader, val_loader, train_sampler, val_sampler, mg_sampler

def build_dataloader_test(cfg):
    """Build dataloader for testing"""
    test_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.VAL_ANNO_PATH,
                                   data_path=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                   mode='test',
                                   clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                   frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                   test_num_segment=cfg.CONFIG.DATA.TEST_NUM_SEGMENT,
                                   test_num_crop=cfg.CONFIG.DATA.TEST_NUM_CROP,
                                   keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                   crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                   short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                   new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                   new_width=cfg.CONFIG.DATA.NEW_WIDTH)

    if cfg.DDP_CONFIG.DISTRIBUTED:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(test_sampler is None),
        num_workers=9, sampler=test_sampler, pin_memory=True)

    return test_loader
