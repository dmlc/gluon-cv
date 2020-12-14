# pylint: disable=line-too-long,too-many-lines,missing-docstring
"""Kinetics700 video action recognition dataset.
Code adapted from https://github.com/open-mmlab/mmaction and
https://github.com/bryanyzhu/two-stream-pytorch"""
import os
from ..video_custom import VideoClsCustom

__all__ = ['Kinetics700']

class Kinetics700(VideoClsCustom):
    """Load the Kinetics700 video action recognition dataset.

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
                 root=os.path.expanduser('~/.mxnet/datasets/kinetics700/rawframes_train'),
                 setting=os.path.expanduser('~/.mxnet/datasets/kinetics700/kinetics700_train_list_rawframes.txt'),
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

        super(Kinetics700, self).__init__(root, setting, train, test_mode, name_pattern,
                                          video_ext, is_color, modality, num_segments,
                                          num_crop, new_length, new_step, new_width, new_height,
                                          target_width, target_height, temporal_jitter,
                                          video_loader, use_decord, slowfast, slow_temporal_stride,
                                          fast_temporal_stride, data_aug, lazy_init, transform)


class Kinetics700Attr(object):
    def __init__(self):
        self.num_class = 700
        self.classes = ['abseiling', 'acting_in_play', 'adjusting_glasses', 'air_drumming', 'alligator_wrestling', 'answering_questions',
                        'applauding', 'applying_cream', 'archaeological_excavation', 'archery', 'arguing', 'arm_wrestling',
                        'arranging_flowers', 'arresting', 'assembling_bicycle', 'assembling_computer', 'attending_conference',
                        'auctioning', 'baby_waking_up', 'backflip_-human-', 'baking_cookies', 'bandaging', 'barbequing', 'bartending',
                        'base_jumping', 'bathing_dog', 'battle_rope_training', 'beatboxing', 'bee_keeping', 'being_excited',
                        'being_in_zero_gravity', 'belly_dancing', 'bench_pressing', 'bending_back', 'bending_metal', 'biking_through_snow',
                        'blasting_sand', 'blending_fruit', 'blowdrying_hair', 'blowing_bubble_gum', 'blowing_glass', 'blowing_leaves',
                        'blowing_nose', 'blowing_out_candles', 'bobsledding', 'bodysurfing', 'bookbinding', 'bottling',
                        'bouncing_ball_-not_juggling-', 'bouncing_on_bouncy_castle', 'bouncing_on_trampoline', 'bowling', 'braiding_hair',
                        'breading_or_breadcrumbing', 'breakdancing', 'breaking_boards', 'breaking_glass', 'breathing_fire',
                        'brush_painting', 'brushing_floor', 'brushing_hair', 'brushing_teeth', 'building_cabinet', 'building_lego',
                        'building_sandcastle', 'building_shed', 'bulldozing', 'bungee_jumping', 'burping', 'busking', 'calculating',
                        'calligraphy', 'canoeing_or_kayaking', 'capoeira', 'capsizing', 'card_stacking', 'card_throwing', 'carrying_baby',
                        'carrying_weight', 'cartwheeling', 'carving_ice', 'carving_marble', 'carving_pumpkin', 'carving_wood_with_a_knife',
                        'casting_fishing_line', 'catching_fish', 'catching_or_throwing_baseball', 'catching_or_throwing_frisbee',
                        'catching_or_throwing_softball', 'celebrating', 'changing_gear_in_car', 'changing_oil',
                        'changing_wheel_-not_on_bike-', 'chasing', 'checking_tires', 'checking_watch', 'cheerleading', 'chewing_gum',
                        'chiseling_stone', 'chiseling_wood', 'chopping_meat', 'chopping_wood', 'clam_digging', 'clapping',
                        'clay_pottery_making', 'clean_and_jerk', 'cleaning_gutters', 'cleaning_pool', 'cleaning_shoes', 'cleaning_toilet',
                        'cleaning_windows', 'climbing_a_rope', 'climbing_ladder', 'climbing_tree', 'closing_door', 'coloring_in',
                        'combing_hair', 'contact_juggling', 'contorting', 'cooking_chicken', 'cooking_egg', 'cooking_on_campfire',
                        'cooking_sausages_-not_on_barbeque-', 'cooking_scallops', 'cosplaying', 'coughing', 'counting_money',
                        'country_line_dancing', 'cracking_back', 'cracking_knuckles', 'cracking_neck', 'crawling_baby', 'crocheting',
                        'crossing_eyes', 'crossing_river', 'crying', 'cumbia', 'curling_-sport-', 'curling_eyelashes', 'curling_hair',
                        'cutting_apple', 'cutting_cake', 'cutting_nails', 'cutting_orange', 'cutting_pineapple', 'cutting_watermelon',
                        'dancing_ballet', 'dancing_charleston', 'dancing_gangnam_style', 'dancing_macarena', 'deadlifting',
                        'dealing_cards', 'decorating_the_christmas_tree', 'decoupage', 'delivering_mail', 'digging', 'dining',
                        'directing_traffic', 'disc_golfing', 'diving_cliff', 'docking_boat', 'dodgeball', 'doing_aerobics',
                        'doing_jigsaw_puzzle', 'doing_laundry', 'doing_nails', 'doing_sudoku', 'drawing', 'dribbling_basketball',
                        'drinking_shots', 'driving_car', 'driving_tractor', 'drooling', 'drop_kicking', 'drumming_fingers',
                        'dumpster_diving', 'dunking_basketball', 'dyeing_eyebrows', 'dyeing_hair', 'eating_burger', 'eating_cake',
                        'eating_carrots', 'eating_chips', 'eating_doughnuts', 'eating_hotdog', 'eating_ice_cream', 'eating_nachos',
                        'eating_spaghetti', 'eating_watermelon', 'egg_hunting', 'embroidering', 'entering_church', 'exercising_arm',
                        'exercising_with_an_exercise_ball', 'extinguishing_fire', 'faceplanting', 'falling_off_bike', 'falling_off_chair',
                        'feeding_birds', 'feeding_fish', 'feeding_goats', 'fencing_-sport-', 'fidgeting', 'filling_cake',
                        'filling_eyebrows', 'finger_snapping', 'fixing_bicycle', 'fixing_hair', 'flint_knapping', 'flipping_bottle',
                        'flipping_pancake', 'fly_tying', 'flying_kite', 'folding_clothes', 'folding_napkins', 'folding_paper',
                        'front_raises', 'frying_vegetables', 'gargling', 'geocaching', 'getting_a_haircut', 'getting_a_piercing',
                        'getting_a_tattoo', 'giving_or_receiving_award', 'gold_panning', 'golf_chipping', 'golf_driving', 'golf_putting',
                        'gospel_singing_in_church', 'grinding_meat', 'grooming_cat', 'grooming_dog', 'grooming_horse',
                        'gymnastics_tumbling', 'hammer_throw', 'hand_washing_clothes', 'head_stand', 'headbanging', 'headbutting',
                        'helmet_diving', 'herding_cattle', 'high_fiving', 'high_jump', 'high_kick', 'historical_reenactment',
                        'hitting_baseball', 'hockey_stop', 'holding_snake', 'home_roasting_coffee', 'hopscotch', 'hoverboarding',
                        'huddling', 'hugging_-not_baby-', 'hugging_baby', 'hula_hooping', 'hurdling', 'hurling_-sport-', 'ice_climbing',
                        'ice_fishing', 'ice_skating', 'ice_swimming', 'inflating_balloons', 'installing_carpet', 'ironing', 'ironing_hair',
                        'javelin_throw', 'jaywalking', 'jetskiing', 'jogging', 'juggling_balls', 'juggling_fire', 'juggling_soccer_ball',
                        'jumping_bicycle', 'jumping_into_pool', 'jumping_jacks', 'jumping_sofa', 'jumpstyle_dancing', 'karaoke',
                        'kicking_field_goal', 'kicking_soccer_ball', 'kissing', 'kitesurfing', 'knitting', 'krumping', 'land_sailing',
                        'laughing', 'lawn_mower_racing', 'laying_bricks', 'laying_concrete', 'laying_decking', 'laying_stone',
                        'laying_tiles', 'leatherworking', 'letting_go_of_balloon', 'licking', 'lifting_hat', 'lighting_candle',
                        'lighting_fire', 'listening_with_headphones', 'lock_picking', 'long_jump', 'longboarding', 'looking_at_phone',
                        'looking_in_mirror', 'luge', 'lunge', 'making_a_cake', 'making_a_sandwich', 'making_balloon_shapes',
                        'making_bubbles', 'making_cheese', 'making_horseshoes', 'making_jewelry', 'making_latte_art',
                        'making_paper_aeroplanes', 'making_pizza', 'making_slime', 'making_snowman', 'making_sushi', 'making_tea',
                        'making_the_bed', 'marching', 'marriage_proposal', 'massaging_back', 'massaging_feet', 'massaging_legs',
                        'massaging_neck', 'massaging_person-s_head', 'metal_detecting', 'milking_cow', 'milking_goat', 'mixing_colours',
                        'moon_walking', 'mopping_floor', 'mosh_pit_dancing', 'motorcycling', 'mountain_climber_-exercise-', 'moving_baby',
                        'moving_child', 'moving_furniture', 'mowing_lawn', 'mushroom_foraging', 'needle_felting', 'news_anchoring',
                        'opening_bottle_-not_wine-', 'opening_coconuts', 'opening_door', 'opening_present', 'opening_refrigerator',
                        'opening_wine_bottle', 'packing', 'paragliding', 'parasailing', 'parkour', 'passing_American_football_-in_game-',
                        'passing_American_football_-not_in_game-', 'passing_soccer_ball', 'peeling_apples', 'peeling_banana',
                        'peeling_potatoes', 'person_collecting_garbage', 'petting_animal_-not_cat-', 'petting_cat', 'petting_horse',
                        'photobombing', 'photocopying', 'picking_apples', 'picking_blueberries', 'pillow_fight', 'pinching',
                        'pirouetting', 'planing_wood', 'planting_trees', 'plastering', 'playing_accordion', 'playing_american_football',
                        'playing_badminton', 'playing_bagpipes', 'playing_basketball', 'playing_bass_guitar', 'playing_beer_pong',
                        'playing_billiards', 'playing_blackjack', 'playing_cards', 'playing_cello', 'playing_checkers', 'playing_chess',
                        'playing_clarinet', 'playing_controller', 'playing_cricket', 'playing_cymbals', 'playing_darts',
                        'playing_didgeridoo', 'playing_dominoes', 'playing_drums', 'playing_field_hockey', 'playing_flute', 'playing_gong',
                        'playing_guitar', 'playing_hand_clapping_games', 'playing_harmonica', 'playing_harp', 'playing_ice_hockey',
                        'playing_keyboard', 'playing_kickball', 'playing_laser_tag', 'playing_lute', 'playing_mahjong', 'playing_maracas',
                        'playing_marbles', 'playing_monopoly', 'playing_netball', 'playing_nose_flute', 'playing_oboe', 'playing_ocarina',
                        'playing_organ', 'playing_paintball', 'playing_pan_pipes', 'playing_piano', 'playing_piccolo', 'playing_pinball',
                        'playing_ping_pong', 'playing_poker', 'playing_polo', 'playing_recorder', 'playing_road_hockey', 'playing_rounders',
                        'playing_rubiks_cube', 'playing_saxophone', 'playing_scrabble', 'playing_shuffleboard', 'playing_slot_machine',
                        'playing_squash_or_racquetball', 'playing_tennis', 'playing_trombone', 'playing_trumpet', 'playing_ukulele',
                        'playing_violin', 'playing_volleyball', 'playing_with_trains', 'playing_xylophone', 'poaching_eggs',
                        'poking_bellybutton', 'pole_vault', 'polishing_furniture', 'polishing_metal', 'popping_balloons', 'pouring_beer',
                        'pouring_milk', 'pouring_wine', 'preparing_salad', 'presenting_weather_forecast', 'pretending_to_be_a_statue',
                        'pull_ups', 'pulling_espresso_shot', 'pulling_rope_-game-', 'pumping_fist', 'pumping_gas', 'punching_bag',
                        'punching_person_-boxing-', 'push_up', 'pushing_car', 'pushing_cart', 'pushing_wheelbarrow', 'pushing_wheelchair',
                        'putting_in_contact_lenses', 'putting_on_eyeliner', 'putting_on_foundation', 'putting_on_lipstick',
                        'putting_on_mascara', 'putting_on_sari', 'putting_on_shoes', 'putting_wallpaper_on_wall', 'raising_eyebrows',
                        'reading_book', 'reading_newspaper', 'recording_music', 'repairing_puncture', 'riding_a_bike', 'riding_camel',
                        'riding_elephant', 'riding_mechanical_bull', 'riding_mule', 'riding_or_walking_with_horse', 'riding_scooter',
                        'riding_snow_blower', 'riding_unicycle', 'ripping_paper', 'roasting_marshmallows', 'roasting_pig',
                        'robot_dancing', 'rock_climbing', 'rock_scissors_paper', 'roller_skating', 'rolling_eyes', 'rolling_pastry',
                        'rope_pushdown', 'running_on_treadmill', 'sailing', 'salsa_dancing', 'saluting', 'sanding_floor', 'sanding_wood',
                        'sausage_making', 'sawing_wood', 'scrambling_eggs', 'scrapbooking', 'scrubbing_face', 'scuba_diving', 'seasoning_food',
                        'separating_eggs', 'setting_table', 'sewing', 'shaking_hands', 'shaking_head', 'shaping_bread_dough',
                        'sharpening_knives', 'sharpening_pencil', 'shaving_head', 'shaving_legs', 'shearing_sheep', 'shining_flashlight',
                        'shining_shoes', 'shoot_dance', 'shooting_basketball', 'shooting_goal_-soccer-', 'shooting_off_fireworks',
                        'shopping', 'shot_put', 'shouting', 'shoveling_snow', 'shredding_paper', 'shucking_oysters', 'shuffling_cards',
                        'shuffling_feet', 'side_kick', 'sieving', 'sign_language_interpreting', 'silent_disco', 'singing', 'sipping_cup',
                        'situp', 'skateboarding', 'ski_ballet', 'ski_jumping', 'skiing_crosscountry', 'skiing_mono', 'skiing_slalom',
                        'skipping_rope', 'skipping_stone', 'skydiving', 'slacklining', 'slapping', 'sled_dog_racing', 'sleeping',
                        'slicing_onion', 'smashing', 'smelling_feet', 'smoking', 'smoking_hookah', 'smoking_pipe', 'snatch_weight_lifting',
                        'sneezing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling', 'somersaulting', 'spelunking',
                        'spinning_plates', 'spinning_poi', 'splashing_water', 'spray_painting', 'spraying', 'springboard_diving',
                        'square_dancing', 'squat', 'squeezing_orange', 'stacking_cups', 'stacking_dice', 'standing_on_hands', 'staring',
                        'steer_roping', 'steering_car', 'sticking_tongue_out', 'stomping_grapes', 'stretching_arm', 'stretching_leg',
                        'sucking_lolly', 'surfing_crowd', 'surfing_water', 'surveying', 'sweeping_floor', 'swimming_backstroke',
                        'swimming_breast_stroke', 'swimming_butterfly_stroke', 'swimming_front_crawl', 'swimming_with_dolphins',
                        'swimming_with_sharks', 'swing_dancing', 'swinging_baseball_bat', 'swinging_on_something', 'sword_fighting',
                        'sword_swallowing', 'tackling', 'tagging_graffiti', 'tai_chi', 'taking_photo', 'talking_on_cell_phone',
                        'tango_dancing', 'tap_dancing', 'tapping_guitar', 'tapping_pen', 'tasting_beer', 'tasting_food', 'tasting_wine',
                        'testifying', 'texting', 'threading_needle', 'throwing_axe', 'throwing_ball_-not_baseball_or_American_football-',
                        'throwing_discus', 'throwing_knife', 'throwing_snowballs', 'throwing_tantrum', 'throwing_water_balloon', 'tickling',
                        'tie_dying', 'tightrope_walking', 'tiptoeing', 'tobogganing', 'tossing_coin', 'tossing_salad', 'training_dog',
                        'trapezing', 'treating_wood', 'trimming_or_shaving_beard', 'trimming_shrubs', 'trimming_trees', 'triple_jump',
                        'twiddling_fingers', 'tying_bow_tie', 'tying_knot_-not_on_a_tie-', 'tying_necktie', 'tying_shoe_laces', 'unboxing',
                        'uncorking_champagne', 'unloading_truck', 'using_a_microscope', 'using_a_paint_roller', 'using_a_power_drill',
                        'using_a_sledge_hammer', 'using_a_wrench', 'using_atm', 'using_bagging_machine', 'using_circular_saw',
                        'using_inhaler', 'using_megaphone', 'using_puppets', 'using_remote_controller_-not_gaming-', 'using_segway',
                        'vacuuming_car', 'vacuuming_floor', 'visiting_the_zoo', 'wading_through_mud', 'wading_through_water',
                        'waiting_in_line', 'waking_up', 'walking_on_stilts', 'walking_the_dog', 'walking_through_snow', 'walking_with_crutches',
                        'washing_dishes', 'washing_feet', 'washing_hair', 'washing_hands', 'watching_tv', 'water_skiing', 'water_sliding',
                        'watering_plants', 'waving_hand', 'waxing_armpits', 'waxing_back', 'waxing_chest', 'waxing_eyebrows', 'waxing_legs',
                        'weaving_basket', 'weaving_fabric', 'welding', 'whistling', 'windsurfing', 'winking', 'wood_burning_-art-',
                        'wrapping_present', 'wrestling', 'writing', 'yarn_spinning', 'yawning', 'yoga', 'zumba']
