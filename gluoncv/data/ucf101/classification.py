# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""UCF101 video action classification dataset.
Code adapted from https://github.com/open-mmlab/mmaction and
https://github.com/bryanyzhu/two-stream-pytorch"""
import os
from ..video_custom import VideoClsCustom

__all__ = ['UCF101']

class UCF101(VideoClsCustom):
    """Load the UCF101 video action recognition dataset.

    Refer to :doc:`../build/examples_datasets/ucf101` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, required. Default '~/.mxnet/datasets/ucf101/rawframes'.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    new_width : int, default 340.
        Scale the width of loaded image to 'new_width' for later multiscale cropping and resizing.
    new_height : int, default 256.
        Scale the height of loaded image to 'new_height' for later multiscale cropping and resizing.
    target_width : int, default 224.
        Scale the width of transformed image to the same 'target_width' for batch forwarding.
    target_height : int, default 224.
        Scale the height of transformed image to the same 'target_height' for batch forwarding.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    slowfast : bool, default False.
        If set to True, use data loader designed for SlowFast network.
        Christoph Feichtenhofer, etal, SlowFast Networks for Video Recognition, ICCV 2019.
    slow_temporal_stride : int, default 16.
        The temporal stride for sparse sampling of video frames in slow branch of a SlowFast network.
    fast_temporal_stride : int, default 2.
        The temporal stride for sparse sampling of video frames in fast branch of a SlowFast network.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
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
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 new_width=340,
                 new_height=256,
                 target_width=224,
                 target_height=224,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 slowfast=False,
                 slow_temporal_stride=16,
                 fast_temporal_stride=2,
                 data_aug='v1',
                 lazy_init=False,
                 transform=None):

        super(UCF101, self).__init__(root, setting, train, test_mode, name_pattern,
                                     video_ext, is_color, modality, num_segments,
                                     num_crop, new_length, new_step, new_width, new_height,
                                     target_width, target_height, temporal_jitter,
                                     video_loader, use_decord, slowfast, slow_temporal_stride,
                                     fast_temporal_stride, data_aug, lazy_init, transform)

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
