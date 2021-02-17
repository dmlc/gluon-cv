
from common.dali_utils import add_dali_pipeline_args, get_rec_pipeline_iter

def add_dali_args(parser):
    return add_dali_pipeline_args(parser, [('--use-dali', 'store_true', 'use dali pipeline and augmentation'),
                                           ('--resize', int, 256, 'Shortest image edge size after resizing')], False)

def get_rec_iter(args, kv=None):
    return get_rec_pipeline_iter(args, kv=kv)

