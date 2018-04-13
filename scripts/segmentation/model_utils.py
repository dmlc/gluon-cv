import os
import importlib

import mxnet as mx
import mxnet.ndarray as F
from mxnet import gluon

from gluonvision.utils.parallel import ModelDataParallel, CriterionDataParallel

__all__ = ['get_model_criterion']

def get_model_criterion(args):
    models = importlib.import_module('gluonvision.model_zoo.'+args.model+'.'+args.model)
    model = models._Net(args.nclass, args.backbone, args.norm_layer)
    print('model', model)
    if args.ignore_index is not None:
        # ignoring boundaries
        criterion = CrossEntropyLoss2d(axis=1, ignore_label=args.ignore_index)
    elif args.bg:
        # ignoring background
        criterion = CrossEntropyLoss2d(axis=1, ignore_label=0)
    else:
        criterion = CrossEntropyLoss2d(axis=1)
    if not args.syncbn and not args.test:
        model.hybridize()
        criterion.hybridize()
    return _init_parallel_model(args, model, criterion)


def _init_parallel_model(args, model, criterion):
    if args.cuda:
        model = ModelDataParallel(model, args.ctx, args.syncbn)
        criterion = CriterionDataParallel(criterion, args.syncbn)
    else:
        model.collect_params().reset_ctx(ctx=args.ctx)
    # resume checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            model.module.load_params(args.resume, ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                .format(args.resume))
    return model, criterion

class CrossEntropyLoss2d(gluon.loss.Loss):
    def __init__(self, axis, ignore_label=None):
        super(CrossEntropyLoss2d, self).__init__(weight=None, batch_axis=0)
        self._axis = axis
        self._ignore_index = ignore_label
 
    def hybrid_forward(self, F, pred, label):
        pred = F.log_softmax(pred, 1)
        loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        if self._ignore_index is not None:
            mask = (label != self._ignore_index).astype(loss.dtype)
            loss = loss * mask.expand_dims(self._axis)
        loss = F.mean(loss, axis=self._batch_axis, exclude=True)
        if self._ignore_index is not None:
            loss = loss * (1.0 * mask.size / mask.sum())
        return loss
