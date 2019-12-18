# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""UCF101 video action classification dataset.
Code partially borrowed from https://github.com/bryanyzhu/two-stream-pytorch"""
import os
import numpy as np
from mxnet import nd
from mxnet.gluon.data import dataset

__all__ = ['UCF101']

class UCF101(dataset.Dataset):
    """Load the UCF101 video action recognition dataset.

    Refer to :doc:`../build/examples_datasets/ucf101` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/ucf101/rawframes'
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
    video_ext : str, default 'mp4'
        If video_loader is set to True, please specify the video format accordinly.
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
    new_step : int, default 1
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    new_width : int, default 340
        Scale the width of loaded image to 'new_width' for later multiscale cropping and resizing.
    new_height : int, default 256
        Scale the height of loaded image to 'new_height' for later multiscale cropping and resizing.
    target_width : int, default 224
        Scale the width of transformed image to the same 'target_width' for batch forwarding.
    target_height : int, default 224
        Scale the height of transformed image to the same 'target_height' for batch forwarding.
    temporal_jitter : bool, default False
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False
        Whether to use video loader to load data.
    use_decord : bool, default True
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes'),
                 setting=os.path.expanduser('~/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'),
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 new_length=1,
                 new_step=1,
                 new_width=340,
                 new_height=256,
                 target_width=224,
                 target_height=224,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 transform=None):

        super(UCF101, self).__init__()

        from ...utils.filesystem import try_import_cv2, try_import_decord, try_import_mmcv
        self.cv2 = try_import_cv2()
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
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform
        self.temporal_jitter = temporal_jitter
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord

        if self.video_loader:
            if self.use_decord:
                self.decord = try_import_decord()
            else:
                self.mmcv = try_import_mmcv()

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
        if self.video_loader:
            if self.use_decord:
                decord_vr = self.decord.VideoReader('{}.{}'.format(directory, self.video_ext), width=self.new_width, height=self.new_height)
                duration = len(decord_vr)
            else:
                mmcv_vr = self.mmcv.VideoReader('{}.{}'.format(directory, self.video_ext))
                duration = len(mmcv_vr)

        if self.train and not self.test_mode:
            segment_indices, skip_offsets = self._sample_train_indices(duration)
        elif not self.train and not self.test_mode:
            segment_indices, skip_offsets = self._sample_val_indices(duration)
        else:
            segment_indices, skip_offsets = self._sample_test_indices(duration)

        # N frames of shape H x W x C, where N = num_oversample * num_segments * new_length
        if self.video_loader:
            if self.use_decord:
                clip_input = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            else:
                clip_input = self._video_TSN_mmcv_loader(directory, mmcv_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = self._image_TSN_cv2_loader(directory, duration, segment_indices, skip_offsets)

        if self.transform is not None:
            clip_input = self.transform(clip_input)

        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (self.new_length, 3, self.target_height, self.target_width))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

        if self.new_length == 1:
            clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

        return nd.array(clip_input), target

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
                if len(line_info) < 3:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(directory, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _sample_val_indices(self, num_frames):
        if num_frames > self.num_segments + self.skip_length - 1:
            tick = (num_frames - self.skip_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _sample_test_indices(self, num_frames):
        if num_frames > self.skip_length - 1:
            tick = (num_frames - self.skip_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _image_TSN_cv2_loader(self, directory, duration, indices, skip_offsets):
        sampled_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_path = os.path.join(directory, self.name_pattern % (offset + skip_offsets[i]))
                else:
                    frame_path = os.path.join(directory, self.name_pattern % (offset))
                cv_img = self.cv2.imread(frame_path)
                if cv_img is None:
                    raise(RuntimeError("Could not load file %s starting at frame %d. Check data path." % (frame_path, offset)))
                if self.new_width > 0 and self.new_height > 0:
                    h, w, _ = cv_img.shape
                    if h != self.new_height or w != self.new_width:
                        cv_img = self.cv2.resize(cv_img, (self.new_width, self.new_height))
                cv_img = cv_img[:, :, ::-1]
                sampled_list.append(cv_img)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return sampled_list

    def _video_TSN_mmcv_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                try:
                    if offset + skip_offsets[i] <= duration:
                        vid_frame = video_reader[offset + skip_offsets[i] - 1]
                    else:
                        vid_frame = video_reader[offset - 1]
                except:
                    raise RuntimeError('Error occured in reading frames from video {} of duration {}.'.format(directory, duration))
                if self.new_width > 0 and self.new_height > 0:
                    h, w, _ = vid_frame.shape
                    if h != self.new_height or w != self.new_width:
                        vid_frame = self.cv2.resize(vid_frame, (self.new_width, self.new_height))
                vid_frame = vid_frame[:, :, ::-1]
                sampled_list.append(vid_frame)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return sampled_list

    def _video_TSN_decord_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                try:
                    if offset + skip_offsets[i] <= duration:
                        vid_frame = video_reader[offset + skip_offsets[i] - 1].asnumpy()
                    else:
                        vid_frame = video_reader[offset - 1].asnumpy()
                except:
                    raise RuntimeError('Error occured in reading frames from video {} of duration {}.'.format(directory, duration))
                sampled_list.append(vid_frame)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return sampled_list

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i]
                else:
                    frame_id = offset
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list

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
