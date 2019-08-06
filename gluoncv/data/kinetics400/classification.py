# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Kinetics400 action classification dataset."""
import os
import random
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
                 setting=os.path.expanduser('~/.mxnet/datasets/kinetics400/kinetics400_train_list_rawframes.txt'),
                 root=os.path.expanduser('~/.mxnet/datasets/kinetics400/rawframes_train'),
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

        super(Kinetics400, self).__init__()

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
                if len(line_info) < 3:
                    print('Video input format is not correct, missing one or more element. %s' % line)
                    continue
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
