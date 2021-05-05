"""Default setting in training/testing for action recognition"""
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
# Maximal number of epochs.
_C.CONFIG.TRAIN.EPOCH_NUM = 196
# Per GPU mini-batch size.
_C.CONFIG.TRAIN.BATCH_SIZE = 8
# Base learning rate.
_C.CONFIG.TRAIN.LR = 0.01
# Momentum.
_C.CONFIG.TRAIN.MOMENTUM = 0.9
# Adam Beta 2
_C.CONFIG.TRAIN.ADAM_BETA2 = 0.98
# Adam eps
_C.CONFIG.TRAIN.ADAM_EPS = 1.5e-09
# L2 regularization.
_C.CONFIG.TRAIN.W_DECAY = 1e-4
# Learning rate policy
_C.CONFIG.TRAIN.LR_POLICY: 'Cosine'
# Steps for multistep learning rate policy
_C.CONFIG.TRAIN.LR_MILESTONE = [40, 80]
# Exponential decay factor.
_C.CONFIG.TRAIN.STEP = 0.1
# Use warm up scheduler or not.
_C.CONFIG.TRAIN.USE_WARMUP: False
# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.CONFIG.TRAIN.WARMUP_EPOCHS: 34
# The start learning rate of the warm up.
_C.CONFIG.TRAIN.WARMUP_START_LR: 0.01
# The end learning rate of the warm up.
_C.CONFIG.TRAIN.WARMUP_END_LR: 0.1
# Resume training from a specific epoch. Set to -1 means train from beginning.
_C.CONFIG.TRAIN.RESUME_EPOCH: -1

# Whether to use multigrid training to speed up.
_C.CONFIG.TRAIN.MULTIGRID = CN(new_allowed=True)
_C.CONFIG.TRAIN.MULTIGRID.USE_LONG_CYCLE = False
_C.CONFIG.TRAIN.MULTIGRID.USE_SHORT_CYCLE = False
_C.CONFIG.TRAIN.MULTIGRID.LONG_CYCLE_EPOCH = [10, 20, 30]

_C.CONFIG.VAL = CN(new_allowed=True)
# Evaluate model on test data every eval period epochs.
_C.CONFIG.VAL.FREQ = 2
# Per GPU mini-batch size.
_C.CONFIG.VAL.BATCH_SIZE = 8


_C.CONFIG.INFERENCE = CN(new_allowed=True)
# Whether to extract features or make predictions.
# If set to True, only features will be returned.
_C.CONFIG.INFERENCE.FEAT = False


_C.CONFIG.DATA = CN(new_allowed=True)

# Paths of annotation files and actual data
_C.CONFIG.DATA.TRAIN_ANNO_PATH = ''
_C.CONFIG.DATA.TRAIN_DATA_PATH = ''
_C.CONFIG.DATA.VAL_ANNO_PATH = ''
_C.CONFIG.DATA.VAL_DATA_PATH = ''
# The number of classes to predict for the model.
_C.CONFIG.DATA.NUM_CLASSES = 400
# The number of frames of the input clip.
_C.CONFIG.DATA.CLIP_LEN = 16
# The video sampling rate of the input clip.
_C.CONFIG.DATA.FRAME_RATE = 2
# Whether to keep aspect ratio when resizing input
_C.CONFIG.DATA.KEEP_ASPECT_RATIO = False
# Temporal segment setting for training video action recognition models.
_C.CONFIG.DATA.NUM_SEGMENT = 1
_C.CONFIG.DATA.NUM_CROP = 1
# Multi-view evaluation for video action recognition models.
# Usually for 2D models, it is 25 segments with 10 crops.
# For 3D models, it is 10 segments with 3 crops.
# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.CONFIG.DATA.TEST_NUM_SEGMENT = 10
# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.CONFIG.DATA.TEST_NUM_CROP = 3
# The spatial crop size for training.
_C.CONFIG.DATA.CROP_SIZE = 224
# Size of the smallest side of the image during testing.
_C.CONFIG.DATA.SHORT_SIDE_SIZE = 256
# Pre-defined height for resizing input video frames.
_C.CONFIG.DATA.NEW_HEIGHT = 256
# Pre-defined width for resizing input video frames.
_C.CONFIG.DATA.NEW_WIDTH = 340
_C.CONFIG.DATA.NUM_WORKERS = 0

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
