"""Default setting in training/testing for directpose"""
from yacs.config import CfgNode as CN


_C = CN()

# ---------------------------------------------------------------------------- #
# Distributed DataParallel setting: DDP_CONFIG
# ---------------------------------------------------------------------------- #

_C.DDP_CONFIG = CN(new_allowed=False)
# Number of nodes for distributed training
_C.DDP_CONFIG.WORLD_SIZE = 1
# Node rank for distributed training
_C.DDP_CONFIG.WORLD_RANK = 0
# Number of GPUs to use
_C.DDP_CONFIG.GPU_WORLD_SIZE = 8
# GPU rank for distributed training
_C.DDP_CONFIG.GPU_WORLD_RANK = 0
# Master node
_C.DDP_CONFIG.DIST_URL = 'tcp://127.0.0.1:10001'
# A list of IP addresses for each node
_C.DDP_CONFIG.WOLRD_URLS = ['127.0.0.1']
# Whether to turn on automatic ranking match.
_C.DDP_CONFIG.AUTO_RANK_MATCH = True
# distributed backend
_C.DDP_CONFIG.DIST_BACKEND = 'nccl'
# Current GPU id.
_C.DDP_CONFIG.GPU = 0
# Whether to use distributed training or simply use dataparallel.
_C.DDP_CONFIG.DISTRIBUTED = True

# ---------------------------------------------------------------------------- #
# Standard training/testing setting: CONFIG
# ---------------------------------------------------------------------------- #

_C.CONFIG = CN(new_allowed=True)

_C.CONFIG.TRAIN = CN(new_allowed=True)
# Per GPU mini-batch size.
_C.CONFIG.TRAIN.BATCH_SIZE = 8
# Base learning rate.
_C.CONFIG.TRAIN.LR = 0.01
# Momentum.
_C.CONFIG.TRAIN.MOMENTUM = 0.9
# L2 regularization.
_C.CONFIG.TRAIN.W_DECAY = 1e-4
# The weight decay that's applied to parameters of normalization layers
_C.CONFIG.TRAIN.W_DECAY_NORM = 0.0
# Learning rate policy
_C.CONFIG.TRAIN.LR_POLICY: 'Cosine'
# Steps for multistep learning rate policy
_C.CONFIG.TRAIN.LR_MILESTONE = [40, 80]
# Exponential decay factor.
_C.CONFIG.TRAIN.STEP = 0.1
# Use warm up scheduler or not.
_C.CONFIG.TRAIN.USE_WARMUP: False
# Gradually warm up the TRAIN.LR over this number of epochs.
_C.CONFIG.TRAIN.WARMUP_EPOCHS: 34
# The LR scheduler name for the iteration based warm up, can be (WarmupMultiStepLR, WarmupCosineLR, WarmupLinearLR)
_C.CONFIG.TRAIN.ITER_LR_SCHEDULER_NAME = "WarmupMultiStepLR"
# Maximal number of epochs.
_C.CONFIG.TRAIN.ITER_NUM = 40000
# The iteration based number to decrease learning rate by SETP, it's similar but different from the epoch based LR_MILESTONE
_C.CONFIG.TRAIN.ITER_LR_STEPS = (30000,)
# Warm up reductio factor for iter based method
_C.CONFIG.TRAIN.ITER_BASED_WARMUP_FACTOR = 1.0 / 1000
# The maximum iterations for the warm up period
_C.CONFIG.TRAIN.ITER_BASED_WARMUP_ITERS = 1000
# The strategy for iteration based warm up
_C.CONFIG.TRAIN.ITER_BASED_WARMUP_METHOD = "linear"
# The start learning rate of the warm up.
_C.CONFIG.TRAIN.WARMUP_START_LR: 0.01
# The end learning rate of the warm up.
_C.CONFIG.TRAIN.WARMUP_END_LR: 0.1

# Gradient clipping
_C.CONFIG.TRAIN.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.CONFIG.TRAIN.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.CONFIG.TRAIN.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.CONFIG.TRAIN.CLIP_GRADIENTS.NORM_TYPE = 2.0


_C.CONFIG.DATA = CN(new_allowed=True)
_C.CONFIG.DATA.NUM_WORKERS = 0
# Switch for loading optional data
_C.CONFIG.DATA.LOAD_PROPOSALS = False
_C.CONFIG.DATA.MASK_ON = False
_C.CONFIG.DATA.MASK_FORMAT = "polygon"  # alternative: "bitmask"
_C.CONFIG.DATA.KEYPOINT_ON = False


# ----------------------------------------------------------------------------
# Registered dataset in registry(torch.data.registry.DatasetCatalog)
# ----------------------------------------------------------------------------
_C.CONFIG.DATA.DATASET = CN(new_allowed=True)
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.CONFIG.DATA.DATASET.TRAIN = ()
# List of the pre-computed proposal files for training, which must be consistent
# with datasets listed in DATASETS.TRAIN.
_C.CONFIG.DATA.DATASET.PROPOSAL_FILES_TRAIN = ()
# Number of top scoring precomputed proposals to keep for training
_C.CONFIG.DATA.DATASET.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.CONFIG.DATA.DATASET.VAL = ()
# List of the pre-computed proposal files for test, which must be consistent
# with datasets listed in DATASETS.TEST.
_C.CONFIG.DATA.DATASET.PROPOSAL_FILES_TEST = ()
# Number of top scoring precomputed proposals to keep for test
_C.CONFIG.DATA.DATASET.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000
_C.CONFIG.DATA.LOAD_PROPOSALS = False


# -----------------------------------------------------------------------------
# Detection Loader
# -----------------------------------------------------------------------------
_C.CONFIG.DATA.DETECTION = CN(new_allowed=True)
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.CONFIG.DATA.DETECTION.ASPECT_RATIO_GROUPING = True
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.CONFIG.DATA.DETECTION.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.CONFIG.DATA.DETECTION.REPEAT_THRESHOLD = 0.0
# Tf True, when working on datasets that have instance annotations, the
# training dataloader will filter out images without associated annotations
_C.CONFIG.DATA.DETECTION.FILTER_EMPTY_ANNOTATIONS = True
# Various Detection Size limitations
_C.CONFIG.DATA.DETECTION.MIN_SIZE_TRAIN_SAMPLING = "choice"
_C.CONFIG.DATA.DETECTION.MIN_SIZE_TRAIN = (800,)
_C.CONFIG.DATA.DETECTION.MAX_SIZE_TRAIN = 1333
_C.CONFIG.DATA.DETECTION.MIN_SIZE_TEST = 800
_C.CONFIG.DATA.DETECTION.MAX_SIZE_TEST = 1333
# `True` if cropping is used for data augmentation during training
_C.CONFIG.DATA.CROP = CN({"ENABLED": False}, new_allowed=True)
# Cropping type:
# - "relative" crop (H * CROP.SIZE[0], W * CROP.SIZE[1]) part of an input of size (H, W)
# - "relative_range" uniformly sample relative crop size from between [CROP.SIZE[0], [CROP.SIZE[1]].
#   and  [1, 1] and use it as in "relative" scenario.
# - "absolute" crop part of an input with absolute size: (CROP.SIZE[0], CROP.SIZE[1]).
# - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
#   [CROP.SIZE[0], min(H, CROP.SIZE[1])] and W_crop in [CROP.SIZE[0], min(W, CROP.SIZE[1])]
_C.CONFIG.DATA.CROP.TYPE = "relative_range"
# Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
# pixels if CROP.TYPE is "absolute"
_C.CONFIG.DATA.CROP.SIZE = [0.9, 0.9]
_C.CONFIG.DATA.CROP.CROP_INSTANCE = True

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

_C.CONFIG.MODEL = CN(new_allowed=True)
# Model architecture. You can find available models in the model zoo.
_C.CONFIG.MODEL.NAME = ''
# Whether to load a checkpoint file. If True, please set the following PRETRAINED_PATH.
_C.CONFIG.MODEL.LOAD = False
# Path (a file path, or URL) to a checkpoint file to be loaded to the model.
_C.CONFIG.MODEL.PRETRAINED_PATH = ''
# Whether to use the trained weights in the model zoo.
_C.CONFIG.MODEL.PRETRAINED = False
# Whether to use pretrained backbone network. Usually this is set to True.
_C.CONFIG.MODEL.PRETRAINED_BASE = True
# BN options
_C.CONFIG.MODEL.BN_EVAL = False
_C.CONFIG.MODEL.PARTIAL_BN = False
_C.CONFIG.MODEL.BN_FROZEN = False
_C.CONFIG.MODEL.USE_AFFINE = False
# TVM trace
_C.CONFIG.MODEL.TVM_MODE = False

# BACKBONE
_C.CONFIG.MODEL.BACKBONE = CN(new_allowed=True)
_C.CONFIG.MODEL.BACKBONE.NAME = "build_directpose_resnet_fpn_backbone"
_C.CONFIG.MODEL.BACKBONE.ANTI_ALIAS = True
_C.CONFIG.MODEL.BACKBONE.FREEZE_AT = 0
_C.CONFIG.MODEL.BACKBONE.LOAD_URL = ''


_C.CONFIG.LOG = CN(new_allowed=True)
# Base directory where all output files are written
_C.CONFIG.LOG.BASE_PATH = ''
# Pre-defined name for each experiment.
# If set to 'use_time', the start time will be appended to the directory name.
_C.CONFIG.LOG.EXP_NAME = 'use_time'
# Directory where training logs are written
_C.CONFIG.LOG.LOG_DIR = 'tb_log'
# Directory where checkpoints are written
_C.CONFIG.LOG.SAVE_DIR = 'checkpoints'
# Directory where testing logs are written
_C.CONFIG.LOG.EVAL_DIR = ''
# Save a checkpoint after every this number of epochs
_C.CONFIG.LOG.SAVE_FREQ = 1
# Display the training log after every this number of iterations
_C.CONFIG.LOG.DISPLAY_FREQ = 1
# LOG LEVEL, choices are (CRITICAL, ERROR, WARNING, INFO, DEBUG)
_C.CONFIG.LOG.LEVEL = 'DEBUG'
# LOG output file, ends with .txt or .log
_C.CONFIG.LOG.LOG_FILENAME = 'log.txt'

# RESNET
_C.CONFIG.MODEL.RESNETS = CN(new_allowed=True)
_C.CONFIG.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
_C.CONFIG.MODEL.RESNETS.DEPTH = 50
_C.CONFIG.MODEL.RESNETS.NORM = "SyncBN"

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.CONFIG.MODEL.DLA = CN()
_C.CONFIG.MODEL.DLA.CONV_BODY = "DLA34"
_C.CONFIG.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.CONFIG.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.CONFIG.MODEL.FPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.CONFIG.MODEL.FPN.IN_FEATURES = []
_C.CONFIG.MODEL.FPN.OUT_CHANNELS = 256

# Options: "" (no norm), "GN"
_C.CONFIG.MODEL.FPN.NORM = ""

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.CONFIG.MODEL.FPN.FUSE_TYPE = "sum"

# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.CONFIG.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.CONFIG.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  # 17 is the number of keypoints in COCO.
# Images with too few (or no) keypoints are excluded from training.
_C.CONFIG.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1

# ---------------------------------------------------------------------------- #
# DIRECTPOSE Head
# ---------------------------------------------------------------------------- #
_C.CONFIG.MODEL.DIRECTPOSE = CN()

# This is the number of foreground classes.
_C.CONFIG.MODEL.DIRECTPOSE.NUM_CLASSES = 1
_C.CONFIG.MODEL.DIRECTPOSE.NUM_KPTS = 17
_C.CONFIG.MODEL.DIRECTPOSE.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.CONFIG.MODEL.DIRECTPOSE.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.CONFIG.MODEL.DIRECTPOSE.PRIOR_PROB = 0.01
_C.CONFIG.MODEL.DIRECTPOSE.INFERENCE_TH_TRAIN = 0.05
_C.CONFIG.MODEL.DIRECTPOSE.INFERENCE_TH_TEST = 0.05
_C.CONFIG.MODEL.DIRECTPOSE.NMS_TH = 0.6
_C.CONFIG.MODEL.DIRECTPOSE.PRE_NMS_TOPK_TRAIN = 1000
_C.CONFIG.MODEL.DIRECTPOSE.PRE_NMS_TOPK_TEST = 1000
_C.CONFIG.MODEL.DIRECTPOSE.POST_NMS_TOPK_TRAIN = 100
_C.CONFIG.MODEL.DIRECTPOSE.POST_NMS_TOPK_TEST = 100
_C.CONFIG.MODEL.DIRECTPOSE.TOP_LEVELS = 2
_C.CONFIG.MODEL.DIRECTPOSE.NORM = "GN"  # Support GN or none
_C.CONFIG.MODEL.DIRECTPOSE.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.CONFIG.MODEL.DIRECTPOSE.THRESH_WITH_CTR = False

# Focal loss parameters for i) Classification and ii) When using Binary Label as heatmap
_C.CONFIG.MODEL.DIRECTPOSE.LOSS_ALPHA = 0.25
_C.CONFIG.MODEL.DIRECTPOSE.LOSS_GAMMA = 2.0
_C.CONFIG.MODEL.DIRECTPOSE.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.CONFIG.MODEL.DIRECTPOSE.USE_RELU = True
_C.CONFIG.MODEL.DIRECTPOSE.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.CONFIG.MODEL.DIRECTPOSE.NUM_CLS_CONVS = 4
_C.CONFIG.MODEL.DIRECTPOSE.NUM_BOX_CONVS = 4
_C.CONFIG.MODEL.DIRECTPOSE.NUM_KPT_CONVS = 4
_C.CONFIG.MODEL.DIRECTPOSE.NUM_HMS_CONVS = 2
_C.CONFIG.MODEL.DIRECTPOSE.NUM_SHARE_CONVS = 0
_C.CONFIG.MODEL.DIRECTPOSE.CENTER_SAMPLE = True
_C.CONFIG.MODEL.DIRECTPOSE.POS_RADIUS = 1.5
_C.CONFIG.MODEL.DIRECTPOSE.LOC_LOSS_TYPE = 'giou'

_C.CONFIG.MODEL.DIRECTPOSE.ENABLE_HM_BRANCH = True
_C.CONFIG.MODEL.DIRECTPOSE.HM_OFFSET = False
_C.CONFIG.MODEL.DIRECTPOSE.HM_CHANNELS = 128
_C.CONFIG.MODEL.DIRECTPOSE.HM_LOSS_TYPE = 'focal'
_C.CONFIG.MODEL.DIRECTPOSE.HM_LOSS_WEIGHT = 4.0
_C.CONFIG.MODEL.DIRECTPOSE.HM_TYPE = 'Gaussian'
_C.CONFIG.MODEL.DIRECTPOSE.HM_MSELOSS_BG_WEIGHT = 1.0      # When using MSE Loss on Gaussian heatmap
_C.CONFIG.MODEL.DIRECTPOSE.HM_MSELOSS_WEIGHT = 1.0
_C.CONFIG.MODEL.DIRECTPOSE.HM_FOCALLOSS_ALPHA = 2          # When using Focal Loss on Gaussian heatmap
_C.CONFIG.MODEL.DIRECTPOSE.HM_FOCALLOSS_BETA = 4
_C.CONFIG.MODEL.DIRECTPOSE.REFINE_KPT = False

_C.CONFIG.MODEL.DIRECTPOSE.KPT_VIS = False
_C.CONFIG.MODEL.DIRECTPOSE.LOSS_ON_LOCATOR = False
_C.CONFIG.MODEL.DIRECTPOSE.KPT_L1_beta = 1/9

_C.CONFIG.MODEL.DIRECTPOSE.ENABLE_BBOX_BRANCH = False
_C.CONFIG.MODEL.DIRECTPOSE.YIELD_PROPOSAL = False
_C.CONFIG.MODEL.DIRECTPOSE.SAMPLE_FEATURE = 'lower'
_C.CONFIG.MODEL.DIRECTPOSE.KPALIGN_GROUPS = 9
_C.CONFIG.MODEL.DIRECTPOSE.SEPERATE_CONV_FEATURE = True
_C.CONFIG.MODEL.DIRECTPOSE.SEPERATE_CONV_CHANNEL = 64

_C.CONFIG.MODEL.DIRECTPOSE.CLOSEKPT_NMS = False
_C.CONFIG.MODEL.DIRECTPOSE.CENTER_BRANCH = 'cls'
