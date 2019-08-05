# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""UCF101 action classification dataset."""
import os
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
    setting : str, required
        Config file of the prepared dataset.
    train : bool, default True
        Whether to load the training or validation set.
    test_mode : bool, default False
        Whether to perform evaluation on the test set
    name_pattern : str, default None
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg
    is_color : bool, default True
        Whether the loaded image is color or grayscale
    modality : str, default 'rgb'
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016
    new_length : int, default 1
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_width : int, default 340
        Scale the width of loaded image to 'new_width' for later multiscale cropping and resizing.
    new_height : int, default 256
        Scale the height of loaded image to 'new_height' for later multiscale cropping and resizing.
    target_width : int, default 224
        Scale the width of transformed image to the same 'target_width' for batch forwarding.
    target_height : int, default 224
        Scale the height of transformed image to the same 'target_height' for batch forwarding.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 setting=os.path.expanduser('~/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_2_rawframes.txt'),
                 root=os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes'),
                 train=True,
                 test_mode=False,
                 name_pattern=None,
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 new_length=1,
                 new_width=340,
                 new_height=256,
                 target_width=224,
                 target_height=224,
                 transform=None):

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
        self.target_height = target_height
        self.target_width = target_width
        self.new_length = new_length
        self.transform = transform

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

        if self.num_segments > 1 and not self.test_mode:
            # For TSN training, reshape the input to B x 3 x H x W. Here, B = batch_size * num_segments
            clip_input = clip_input.reshape((-1, 3 * self.new_length, self.target_height, self.target_width))

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
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                # line format: video_path, video_duration, video_label
                clip_path = os.path.join(directory, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
        return clips

    def _TSN_RGB(self, directory, offsets, new_height, new_width, new_length, is_color, name_pattern):

        from ...utils.filesystem import try_import_cv2
        cv2 = try_import_cv2()

        if is_color:
            cv_read_flag = cv2.IMREAD_COLOR
        else:
            cv_read_flag = cv2.IMREAD_GRAYSCALE
        interpolation = cv2.INTER_LINEAR

        sampled_list = []
        for _, offset in enumerate(offsets):
            for length_id in range(1, new_length+1):
                frame_name = name_pattern % (length_id + offset)
                frame_path = directory + "/" + frame_name
                cv_img_origin = cv2.imread(frame_path, cv_read_flag)
                if cv_img_origin is None:
                    raise(RuntimeError("Could not load file %s. Check data path." % (frame_path)))
                if new_width > 0 and new_height > 0:
                    cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
                else:
                    cv_img = cv_img_origin
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                sampled_list.append(cv_img)
        # the shape of clip_input will be H x W x C, and C = num_segments * new_length * 3
        clip_input = np.concatenate(sampled_list, axis=2)
        return nd.array(clip_input)

class UCF101Attr(object):
    def __init__(self):
        self.num_class = 101
        self.classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
                        'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
                        'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
                        'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
                        'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
                        'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch',
                        'FrontCrawl', 'GolfSwing', 'Haircut', 'HammerThrow', 'Hammering', 'HandstandPushups',
                        'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop',
                        'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking',
                        'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks',
                        'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute',
                        'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin',
                        'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor',
                        'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding',
                        'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings',
                        'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing',
                        'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking',
                        'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']
