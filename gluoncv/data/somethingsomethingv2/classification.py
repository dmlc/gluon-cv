# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Something-something-v2 action classification dataset."""
import os
import numpy as np
from mxnet import nd
from mxnet.gluon.data import dataset

__all__ = ['SomethingSomethingV2']

class SomethingSomethingV2(dataset.Dataset):
    """Load the something-something-v2 action recognition dataset.

    Refer to :doc:`../build/examples_datasets/somethingsomethingv2` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2-frames'
        Path to the folder stored the dataset.
    setting : str, required
        Config file of the prepared dataset.
    train : bool, default True
        Whether to load the training or validation set.
    test_mode : bool, default False
        Whether to perform evaluation on the test set
    name_pattern : str, default None
        The naming pattern of the decoded video frames.
        For example, 000012.jpg
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
                 root=os.path.expanduser('~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2-frames'),
                 setting=os.path.expanduser('~/.mxnet/datasets/somethingsomethingv2/train_videofolder.txt'),
                 train=True,
                 test_mode=False,
                 name_pattern='%06d.jpg',
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

        super(SomethingSomethingV2, self).__init__()

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
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord

        if self.video_loader:
            if self.use_decord:
                self.decord = try_import_decord()
            else:
                self.mmcv = try_import_mmcv()

        self.clips = self._make_dataset(root, setting)
        if len(self.clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory (opt.data-dir)."))

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
