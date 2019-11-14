from __future__ import print_function

import numpy as np
import gluoncv as gcv
from gluoncv.model_zoo.model_store import pretrained_model_list
from common import try_gpu

@try_gpu(0)
def test_export_model_zoo():
    for model in pretrained_model_list():
        print('exporting:', model)

        if 'deeplab' in model or 'psp' in model:
            # semantic segmentation models require fixed data shape
            kwargs = {'data_shape':(480, 480, 3)}
        elif '3d' in model:
            # video action recognition models require 4d data shape
            kwargs = {'data_shape':(3, 32, 224, 224), 'layout':'CTHW', 'preprocess':None}
        elif 'slowfast_4x16' in model:
            # video action recognition models require 4d data shape
            kwargs = {'data_shape':(3, 36, 224, 224), 'layout':'CTHW', 'preprocess':None}
        elif 'slowfast_8x8' in model:
            # video action recognition models require 4d data shape
            kwargs = {'data_shape':(3, 40, 224, 224), 'layout':'CTHW', 'preprocess':None}
        else:
            kwargs = {}

        if '_gn' in model:
            continue

        try:
            gcv.utils.export_block(model, gcv.model_zoo.get_model(model, pretrained=True), **kwargs)
        except ValueError:
            # ignore non defined model name
            pass
        except AttributeError:
            # deeplab model do not support it now, skip
            pass

if __name__ == '__main__':
    import nose
    nose.runmodule()
