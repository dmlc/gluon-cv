# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Kinetics400 video action recognition dataset.
Code adapted from https://github.com/open-mmlab/mmaction and
https://github.com/bryanyzhu/two-stream-pytorch"""
import os
from ..video_custom import VideoClsCustom

__all__ = ['Kinetics400']

class Kinetics400(VideoClsCustom):
    """Load the Kinetics400 video action recognition dataset.

    Refer to :doc:`../build/examples_datasets/kinetics400` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, required. Default '~/.mxnet/datasets/kinetics400/rawframes_train'.
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
                 data_aug='v1',
                 lazy_init=False,
                 transform=None):

        super(Kinetics400, self).__init__(root, setting, train, test_mode, name_pattern,
                                          video_ext, is_color, modality, num_segments,
                                          num_crop, new_length, new_step, new_width, new_height,
                                          target_width, target_height, temporal_jitter,
                                          video_loader, use_decord, slowfast, slow_temporal_stride,
                                          fast_temporal_stride, data_aug, lazy_init, transform)

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
