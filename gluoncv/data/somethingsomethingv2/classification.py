# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Something-something-v2 video action classification dataset.
Code adapted from https://github.com/open-mmlab/mmaction and
https://github.com/bryanyzhu/two-stream-pytorch"""
import os
from ..video_custom import VideoClsCustom

__all__ = ['SomethingSomethingV2']

class SomethingSomethingV2(VideoClsCustom):
    """Load the Something-Something-V2 video action recognition dataset.

    Refer to :doc:`../build/examples_datasets/somethingsomethingv2` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, required. Default '~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2-frames'.
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
        For example, 000012.jpg.
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
        Different types of data augmentation pipelines. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 root=os.path.expanduser('~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2-frames'),
                 setting=os.path.expanduser('~/.mxnet/datasets/somethingsomethingv2/train_videofolder.txt'),
                 train=True,
                 test_mode=False,
                 name_pattern='%06d.jpg',
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

        super(SomethingSomethingV2, self).__init__(root, setting, train, test_mode, name_pattern,
                                                   video_ext, is_color, modality, num_segments,
                                                   num_crop, new_length, new_step, new_width, new_height,
                                                   target_width, target_height, temporal_jitter,
                                                   video_loader, use_decord, slowfast, slow_temporal_stride,
                                                   fast_temporal_stride, data_aug, lazy_init, transform)

class SomethingSomethingV2Attr(object):
    def __init__(self):
        self.num_class = 174
        self.classes = ['Approaching something with your camera', 'Attaching something to something',
                        'Bending something so that it deforms', 'Bending something until it breaks',
                        'Burying something in something', 'Closing something', 'Covering something with something',
                        'Digging something out of something', 'Dropping something behind something',
                        'Dropping something in front of something', 'Dropping something into something',
                        'Dropping something next to something', 'Dropping something onto something',
                        'Failing to put something into something because something does not fit',
                        'Folding something', 'Hitting something with something', 'Holding something',
                        'Holding something behind something', 'Holding something in front of something',
                        'Holding something next to something', 'Holding something over something',
                        'Laying something on the table on its side, not upright', 'Letting something roll along a flat surface',
                        'Letting something roll down a slanted surface', 'Letting something roll up a slanted surface, so it rolls back down',
                        'Lifting a surface with something on it but not enough for it to slide down',
                        'Lifting a surface with something on it until it starts sliding down',
                        'Lifting something up completely without letting it drop down',
                        'Lifting something up completely, then letting it drop down', 'Lifting something with something on it',
                        'Lifting up one end of something without letting it drop down',
                        'Lifting up one end of something, then letting it drop down', 'Moving away from something with your camera',
                        'Moving part of something', 'Moving something across a surface until it falls down',
                        'Moving something across a surface without it falling down', 'Moving something and something away from each other',
                        'Moving something and something closer to each other', 'Moving something and something so they collide with each other',
                        'Moving something and something so they pass each other', 'Moving something away from something',
                        'Moving something away from the camera', 'Moving something closer to something', 'Moving something down',
                        'Moving something towards the camera', 'Moving something up', 'Opening something', 'Picking something up',
                        'Piling something up', 'Plugging something into something',
                        'Plugging something into something but pulling it right out as you remove your hand',
                        'Poking a hole into some substance', 'Poking a hole into something soft',
                        'Poking a stack of something so the stack collapses', 'Poking a stack of something without the stack collapsing',
                        'Poking something so it slightly moves', "Poking something so lightly that it doesn't or almost doesn't move",
                        'Poking something so that it falls over', 'Poking something so that it spins around',
                        'Pouring something into something', 'Pouring something into something until it overflows',
                        'Pouring something onto something', 'Pouring something out of something',
                        'Pretending or failing to wipe something off of something', 'Pretending or trying and failing to twist something',
                        'Pretending to be tearing something that is not tearable', 'Pretending to close something without actually closing it',
                        'Pretending to open something without actually opening it', 'Pretending to pick something up',
                        'Pretending to poke something', 'Pretending to pour something out of something, but something is empty',
                        'Pretending to put something behind something', 'Pretending to put something into something',
                        'Pretending to put something next to something', 'Pretending to put something on a surface',
                        'Pretending to put something onto something', 'Pretending to put something underneath something',
                        'Pretending to scoop something up with something', 'Pretending to spread air onto something',
                        'Pretending to sprinkle air onto something', 'Pretending to squeeze something', 'Pretending to take something from somewhere',
                        'Pretending to take something out of something', 'Pretending to throw something',
                        'Pretending to turn something upside down', 'Pulling something from behind of something',
                        'Pulling something from left to right', 'Pulling something from right to left', 'Pulling something onto something',
                        'Pulling something out of something', 'Pulling two ends of something but nothing happens',
                        'Pulling two ends of something so that it gets stretched',
                        'Pulling two ends of something so that it separates into two pieces',
                        'Pushing something from left to right', 'Pushing something from right to left',
                        'Pushing something off of something', 'Pushing something onto something', 'Pushing something so it spins',
                        "Pushing something so that it almost falls off but doesn't", 'Pushing something so that it falls off the table',
                        'Pushing something so that it slightly moves', 'Pushing something with something',
                        'Putting number of something onto something', 'Putting something and something on the table',
                        'Putting something behind something', 'Putting something in front of something', 'Putting something into something',
                        'Putting something next to something', 'Putting something on a flat surface without letting it roll',
                        'Putting something on a surface', 'Putting something on the edge of something so it is not supported and falls down',
                        "Putting something onto a slanted surface but it doesn't glide down", 'Putting something onto something',
                        'Putting something onto something else that cannot support it so it falls down',
                        'Putting something similar to other things that are already on the table',
                        "Putting something that can't roll onto a slanted surface, so it slides down",
                        "Putting something that can't roll onto a slanted surface, so it stays where it is",
                        'Putting something that cannot actually stand upright upright on the table, so it falls on its side',
                        'Putting something underneath something', 'Putting something upright on the table',
                        'Putting something, something and something on the table', 'Removing something, revealing something behind',
                        'Rolling something on a flat surface', 'Scooping something up with something',
                        'Showing a photo of something to the camera', 'Showing something behind something',
                        'Showing something next to something', 'Showing something on top of something', 'Showing something to the camera',
                        'Showing that something is empty', 'Showing that something is inside something',
                        'Something being deflected from something', 'Something colliding with something and both are being deflected',
                        'Something colliding with something and both come to a halt', 'Something falling like a feather or paper',
                        'Something falling like a rock', 'Spilling something behind something', 'Spilling something next to something',
                        'Spilling something onto something', 'Spinning something so it continues spinning',
                        'Spinning something that quickly stops spinning', 'Spreading something onto something',
                        'Sprinkling something onto something', 'Squeezing something', 'Stacking number of something',
                        'Stuffing something into something', 'Taking one of many similar things on the table',
                        'Taking something from somewhere', 'Taking something out of something', 'Tearing something into two pieces',
                        'Tearing something just a little bit', 'Throwing something', 'Throwing something against something',
                        'Throwing something in the air and catching it', 'Throwing something in the air and letting it fall',
                        'Throwing something onto a surface', "Tilting something with something on it slightly so it doesn't fall down",
                        'Tilting something with something on it until it falls off', 'Tipping something over',
                        'Tipping something with something in it over, so something in it falls out',
                        'Touching (without moving) part of something', "Trying but failing to attach something to something because it doesn't stick",
                        'Trying to bend something unbendable so nothing happens', 'Trying to pour something into something, but missing so it spills next to it',
                        'Turning something upside down', 'Turning the camera downwards while filming something',
                        'Turning the camera left while filming something', 'Turning the camera right while filming something',
                        'Turning the camera upwards while filming something', 'Twisting (wringing) something wet until water comes out',
                        'Twisting something', 'Uncovering something', 'Unfolding something', 'Wiping something off of something']
