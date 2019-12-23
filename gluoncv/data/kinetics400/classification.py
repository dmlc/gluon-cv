# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Kinetics400 action classification dataset.
Code partially borrowed from https://github.com/open-mmlab/mmaction.
Code partially borrowed from https://github.com/bryanyzhu/two-stream-pytorch"""
import os
import numpy as np
from mxnet import nd
from mxnet.gluon.data import dataset

__all__ = ['Kinetics400']

class Kinetics400(dataset.Dataset):
    """Load the Kinetics400 action recognition dataset.

    Refer to :doc:`../build/examples_datasets/kinetics400` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/kinetics400'
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
    num_crop : int, default 1
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
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
                 root=os.path.expanduser('~/.mxnet/datasets/kinetics400/rawframes_train'),
                 setting=os.path.expanduser('~/.mxnet/datasets/kinetics400/kinetics400_train_list_rawframes.txt'),
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
                 transform=None):

        super(Kinetics400, self).__init__()

        from ...utils.filesystem import try_import_cv2, try_import_decord, try_import_mmcv
        self.cv2 = try_import_cv2()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
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
        self.slowfast = slowfast
        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride

        if self.slowfast:
            assert slow_temporal_stride % fast_temporal_stride == 0, 'slow_temporal_stride needs to be multiples of slow_temporal_stride, please set it accordinly.'
            assert not temporal_jitter, 'Slowfast dataloader does not support temporal jitter. Please set temporal_jitter=False.'
            assert new_step == 1, 'Slowfast dataloader only support consecutive frames reading, please set new_step=1.'

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

    def __getitem__(self, index):

        directory, duration, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)
            if self.use_decord:
                decord_vr = self.decord.VideoReader(video_name, width=self.new_width, height=self.new_height)
                duration = len(decord_vr)
            else:
                mmcv_vr = self.mmcv.VideoReader(video_name)
                duration = len(mmcv_vr)

        if self.train and not self.test_mode:
            segment_indices, skip_offsets = self._sample_train_indices(duration)
        elif not self.train and not self.test_mode:
            segment_indices, skip_offsets = self._sample_val_indices(duration)
        else:
            segment_indices, skip_offsets = self._sample_test_indices(duration)

        # N frames of shape H x W x C, where N = num_oversample * num_segments * new_length
        if self.video_loader:
            if self.slowfast:
                clip_input = self._video_TSN_decord_slowfast_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            else:
                if self.use_decord:
                    clip_input = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
                else:
                    clip_input = self._video_TSN_mmcv_loader(directory, mmcv_vr, duration, segment_indices, skip_offsets)
        else:
            if self.slowfast:
                clip_input = self._image_slowfast_cv2_loader(directory, duration, segment_indices, skip_offsets)
            else:
                clip_input = self._image_TSN_cv2_loader(directory, duration, segment_indices, skip_offsets)

        if self.transform is not None:
            clip_input = self.transform(clip_input)

        if self.slowfast:
            sparse_sampels = len(clip_input) // (self.num_segments * self.num_crop)
            clip_input = np.stack(clip_input, axis=0)
            clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, self.target_height, self.target_width))
            clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        else:
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

    def _image_slowfast_cv2_loader(self, directory, duration, indices, skip_offsets):
        sampled_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            fast_list = []
            slow_list = []
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_path = os.path.join(directory, self.name_pattern % (offset + skip_offsets[i]))
                else:
                    frame_path = os.path.join(directory, self.name_pattern % (offset))
                if (i + 1) % self.fast_temporal_stride == 0:
                    cv_img = self.cv2.imread(frame_path)
                    if cv_img is None:
                        raise(RuntimeError("Could not load file %s starting at frame %d. Check data path." % (frame_path, offset)))
                    if self.new_width > 0 and self.new_height > 0:
                        h, w, _ = cv_img.shape
                        if h != self.new_height or w != self.new_width:
                            cv_img = self.cv2.resize(cv_img, (self.new_width, self.new_height))
                    cv_img = cv_img[:, :, ::-1]
                    fast_list.append(cv_img)

                    if (i + 1) % self.slow_temporal_stride == 0:
                        slow_list.append(cv_img)

                if offset + self.new_step < duration:
                    offset += self.new_step
            fast_list.extend(slow_list)
            sampled_list.extend(fast_list)
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
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list

    def _video_TSN_decord_slowfast_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            fast_id_list = []
            slow_id_list = []
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1

                if (i + 1) % self.fast_temporal_stride == 0:
                    fast_id_list.append(frame_id)

                    if (i + 1) % self.slow_temporal_stride == 0:
                        slow_id_list.append(frame_id)

                if offset + self.new_step < duration:
                    offset += self.new_step

            fast_id_list.extend(slow_id_list)
            frame_id_list.extend(fast_id_list)
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list

class Kinetics400Attr(object):
    def __init__(self):
        self.num_class = 400
        self.classes = ['abseiling', 'air_drumming', 'answering_questions', 'applauding', 'applying_cream', 'archery',
                        'arm_wrestling', 'arranging_flowers', 'assembling_computer', 'auctioning', 'baby_waking_up', 'baking_cookies',
                        'balloon_blowing', 'bandaging', 'barbequing', 'bartending', 'beatboxing', 'bee_keeping', 'belly_dancing',
                        'bench_pressing', 'bending_back', 'bending_metal', 'biking_through_snow', 'blasting_sand', 'blowing_glass',
                        'blowing_leaves', 'blowing_nose', 'blowing_out_candles', 'bobsledding', 'bookbinding', 'bouncing_on_trampoline',
                        'bowling', 'braiding_hair', 'breading_or_breadcrumbing', 'breakdancing', 'brush_painting', 'brushing_hair',
                        'brushing_teeth', 'building_cabinet', 'building_shed', 'bungee_jumping', 'busking', 'canoeing_or_kayaking',
                        'capoeira', 'carrying_baby', 'cartwheeling', 'carving_pumpkin', 'catching_fish', 'catching_or_throwing_baseball',
                        'catching_or_throwing_frisbee', 'catching_or_throwing_softball', 'celebrating', 'changing_oil', 'changing_wheel',
                        'checking_tires', 'cheerleading', 'chopping_wood', 'clapping', 'clay_pottery_making', 'clean_and_jerk',
                        'cleaning_floor', 'cleaning_gutters', 'cleaning_pool', 'cleaning_shoes', 'cleaning_toilet', 'cleaning_windows',
                        'climbing_a_rope', 'climbing_ladder', 'climbing_tree', 'contact_juggling', 'cooking_chicken', 'cooking_egg',
                        'cooking_on_campfire', 'cooking_sausages', 'counting_money', 'country_line_dancing', 'cracking_neck', 'crawling_baby',
                        'crossing_river', 'crying', 'curling_hair', 'cutting_nails', 'cutting_pineapple', 'cutting_watermelon',
                        'dancing_ballet', 'dancing_charleston', 'dancing_gangnam_style', 'dancing_macarena', 'deadlifting',
                        'decorating_the_christmas_tree', 'digging', 'dining', 'disc_golfing', 'diving_cliff', 'dodgeball', 'doing_aerobics',
                        'doing_laundry', 'doing_nails', 'drawing', 'dribbling_basketball', 'drinking', 'drinking_beer', 'drinking_shots',
                        'driving_car', 'driving_tractor', 'drop_kicking', 'drumming_fingers', 'dunking_basketball', 'dying_hair',
                        'eating_burger', 'eating_cake', 'eating_carrots', 'eating_chips', 'eating_doughnuts', 'eating_hotdog',
                        'eating_ice_cream', 'eating_spaghetti', 'eating_watermelon', 'egg_hunting', 'exercising_arm',
                        'exercising_with_an_exercise_ball', 'extinguishing_fire', 'faceplanting', 'feeding_birds', 'feeding_fish',
                        'feeding_goats', 'filling_eyebrows', 'finger_snapping', 'fixing_hair', 'flipping_pancake', 'flying_kite',
                        'folding_clothes', 'folding_napkins', 'folding_paper', 'front_raises', 'frying_vegetables', 'garbage_collecting',
                        'gargling', 'getting_a_haircut', 'getting_a_tattoo', 'giving_or_receiving_award', 'golf_chipping', 'golf_driving',
                        'golf_putting', 'grinding_meat', 'grooming_dog', 'grooming_horse', 'gymnastics_tumbling', 'hammer_throw',
                        'headbanging', 'headbutting', 'high_jump', 'high_kick', 'hitting_baseball', 'hockey_stop', 'holding_snake',
                        'hopscotch', 'hoverboarding', 'hugging', 'hula_hooping', 'hurdling', 'hurling_-sport-', 'ice_climbing', 'ice_fishing',
                        'ice_skating', 'ironing', 'javelin_throw', 'jetskiing', 'jogging', 'juggling_balls', 'juggling_fire',
                        'juggling_soccer_ball', 'jumping_into_pool', 'jumpstyle_dancing', 'kicking_field_goal', 'kicking_soccer_ball',
                        'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing', 'laying_bricks', 'long_jump', 'lunge', 'making_a_cake',
                        'making_a_sandwich', 'making_bed', 'making_jewelry', 'making_pizza', 'making_snowman', 'making_sushi', 'making_tea',
                        'marching', 'massaging_back', 'massaging_feet', 'massaging_legs', "massaging_person's_head", 'milking_cow',
                        'mopping_floor', 'motorcycling', 'moving_furniture', 'mowing_lawn', 'news_anchoring', 'opening_bottle',
                        'opening_present', 'paragliding', 'parasailing', 'parkour', 'passing_American_football_-in_game-',
                        'passing_American_football_-not_in_game-', 'peeling_apples', 'peeling_potatoes', 'petting_animal_-not_cat-',
                        'petting_cat', 'picking_fruit', 'planting_trees', 'plastering', 'playing_accordion', 'playing_badminton',
                        'playing_bagpipes', 'playing_basketball', 'playing_bass_guitar', 'playing_cards', 'playing_cello', 'playing_chess',
                        'playing_clarinet', 'playing_controller', 'playing_cricket', 'playing_cymbals', 'playing_didgeridoo', 'playing_drums',
                        'playing_flute', 'playing_guitar', 'playing_harmonica', 'playing_harp', 'playing_ice_hockey', 'playing_keyboard',
                        'playing_kickball', 'playing_monopoly', 'playing_organ', 'playing_paintball', 'playing_piano', 'playing_poker',
                        'playing_recorder', 'playing_saxophone', 'playing_squash_or_racquetball', 'playing_tennis', 'playing_trombone',
                        'playing_trumpet', 'playing_ukulele', 'playing_violin', 'playing_volleyball', 'playing_xylophone', 'pole_vault',
                        'presenting_weather_forecast', 'pull_ups', 'pumping_fist', 'pumping_gas', 'punching_bag', 'punching_person_-boxing-',
                        'push_up', 'pushing_car', 'pushing_cart', 'pushing_wheelchair', 'reading_book', 'reading_newspaper', 'recording_music',
                        'riding_a_bike', 'riding_camel', 'riding_elephant', 'riding_mechanical_bull', 'riding_mountain_bike', 'riding_mule',
                        'riding_or_walking_with_horse', 'riding_scooter', 'riding_unicycle', 'ripping_paper', 'robot_dancing', 'rock_climbing',
                        'rock_scissors_paper', 'roller_skating', 'running_on_treadmill', 'sailing', 'salsa_dancing', 'sanding_floor',
                        'scrambling_eggs', 'scuba_diving', 'setting_table', 'shaking_hands', 'shaking_head', 'sharpening_knives',
                        'sharpening_pencil', 'shaving_head', 'shaving_legs', 'shearing_sheep', 'shining_shoes', 'shooting_basketball',
                        'shooting_goal_-soccer-', 'shot_put', 'shoveling_snow', 'shredding_paper', 'shuffling_cards', 'side_kick',
                        'sign_language_interpreting', 'singing', 'situp', 'skateboarding', 'ski_jumping', 'skiing_-not_slalom_or_crosscountry-',
                        'skiing_crosscountry', 'skiing_slalom', 'skipping_rope', 'skydiving', 'slacklining', 'slapping', 'sled_dog_racing',
                        'smoking', 'smoking_hookah', 'snatch_weight_lifting', 'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting',
                        'snowmobiling', 'somersaulting', 'spinning_poi', 'spray_painting', 'spraying', 'springboard_diving', 'squat',
                        'sticking_tongue_out', 'stomping_grapes', 'stretching_arm', 'stretching_leg', 'strumming_guitar', 'surfing_crowd',
                        'surfing_water', 'sweeping_floor', 'swimming_backstroke', 'swimming_breast_stroke', 'swimming_butterfly_stroke',
                        'swing_dancing', 'swinging_legs', 'swinging_on_something', 'sword_fighting', 'tai_chi', 'taking_a_shower', 'tango_dancing',
                        'tap_dancing', 'tapping_guitar', 'tapping_pen', 'tasting_beer', 'tasting_food', 'testifying', 'texting', 'throwing_axe',
                        'throwing_ball', 'throwing_discus', 'tickling', 'tobogganing', 'tossing_coin', 'tossing_salad', 'training_dog',
                        'trapezing', 'trimming_or_shaving_beard', 'trimming_trees', 'triple_jump', 'tying_bow_tie', 'tying_knot_-not_on_a_tie-',
                        'tying_tie', 'unboxing', 'unloading_truck', 'using_computer', 'using_remote_controller_-not_gaming-', 'using_segway',
                        'vault', 'waiting_in_line', 'walking_the_dog', 'washing_dishes', 'washing_feet', 'washing_hair', 'washing_hands',
                        'water_skiing', 'water_sliding', 'watering_plants', 'waxing_back', 'waxing_chest', 'waxing_eyebrows', 'waxing_legs',
                        'weaving_basket', 'welding', 'whistling', 'windsurfing', 'wrapping_present', 'wrestling', 'writing', 'yawning', 'yoga', 'zumba']
