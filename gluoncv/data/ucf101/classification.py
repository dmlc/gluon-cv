# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""UCF101 action classification dataset."""
import os, sys
import cv2
import random
import numpy as np
from mxnet import nd
from mxnet.gluon.data import dataset

__all__ = ['UCF101']

class UCF101(dataset.Dataset):
    """Load the UCF101 action recognition dataset.

    Refer to :doc:`../build/examples_datasets/ucf101` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/ucf101'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    transform : function, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples. (TODO, should we restrict its datatype
        to transformer?)
    """
    def __init__(self, 
                 setting, 
                 root=os.path.join('~', '.mxnet', 'datasets', 'ucf101'),
                 train=True, 
                 test_mode=False,
                 name_pattern=None, 
                 is_color=True, 
                 modality='rgb', 
                 num_segments=1, 
                 new_length=1, 
                 new_width=340, 
                 new_height=256, 
                 transform=None, 
                 target_transform=None, 
                 video_transform=None):

        super(UCF101, self).__init__()

        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.new_height = new_height
        self.new_width = new_width
        self.new_length = new_length
        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

        self.classes, self.class_to_idx = self._find_classes(root)
        self.clips = self._make_dataset(root, setting)
        if len(self.clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory (opt.data-dir)."))

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "img_%05d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d.jpg"

    def __getitem__(self, index):

        directory, duration, target = self.clips[index]
        average_duration = int(duration / self.num_segments)
        offsets = []
        for seg_id in range(self.num_segments):
            if self.train and not self.test_mode:
                # training
                if average_duration >= self.new_length:
                    offset = random.randint(0, average_duration - self.new_length)
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                else:
                    offsets.append(0)
            elif not self.train and not self.test_mode:
                # validation
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                else:
                    offsets.append(0)
            else:
                # test
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                else:
                    offsets.append(0)
        
        clip_input = self._TSN_RGB(directory, offsets, self.new_height, self.new_width, self.new_length, self.is_color, self.name_pattern)
        
        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        return clip_input, target

    def __len__(self):
        return len(self.clips)

    def _find_classes(self, directory):

        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, directory, setting):

        if not os.path.exists(setting):
            print("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting))
            sys.exit()
        else:
            clips = []
            with open(setting) as split_f:
                data = split_f.readlines()
                for line in data:
                    line_info = line.split()
                    clip_path = os.path.join(directory, line_info[0])
                    duration = int(line_info[1])
                    target = int(line_info[2])
                    item = (clip_path, duration, target)
                    clips.append(item)
        return clips

    def _TSN_RGB(self, directory, offsets, new_height, new_width, new_length, is_color, name_pattern):
        if is_color:
            cv_read_flag = cv2.IMREAD_COLOR         # > 0
        else:
            cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
        interpolation = cv2.INTER_LINEAR

        sampled_list = []
        for offset_id in range(len(offsets)):
            offset = offsets[offset_id]
            for length_id in range(1, new_length+1):
                frame_name = name_pattern % (length_id + offset)
                frame_path = directory + "/" + frame_name
                cv_img_origin = cv2.imread(frame_path, cv_read_flag)
                if cv_img_origin is None:
                   print("Could not load file %s" % (frame_path))
                   sys.exit()
                   # TODO: error handling here
                if new_width > 0 and new_height > 0:
                    # use OpenCV3, use OpenCV2.4.13 may have error
                    cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
                else:
                    cv_img = cv_img_origin
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                sampled_list.append(cv_img)
        clip_input = np.concatenate(sampled_list, axis=2)
        return nd.array(clip_input)


