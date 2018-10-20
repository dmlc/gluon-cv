from __future__ import print_function

import numpy as np
import gluoncv as gcv
from gluoncv.model_zoo.model_store import pretrained_model_list

def test_export_model_zoo():
    for model in pretrained_model_list():
        print('exporting:', model)
        try:
            gcv.utils.export_block(model, gcv.model_zoo.get_model(model, pretrained=True))
        except:
            # ignore non defined model name
            pass

if __name__ == '__main__':
    import nose
    nose.runmodule()
