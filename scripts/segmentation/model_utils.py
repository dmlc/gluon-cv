import os
import importlib

import mxnet as mx
import mxnet.ndarray as F
from mxnet import gluon

from gluonvision.utils import ModelDataParallel, CriterionDataParallel
from gluonvision.model_zoo.losses import SoftmaxCrossEntropyLoss

__all__ = ['get_model_criterion']

def get_model_criterion(args):
    models = importlib.import_module('gluonvision.model_zoo.'+args.model+'.'+args.model)
    model = models._Net(args)
    print('model', model)
    if args.ignore_index is not None:
        # ignoring boundaries
        criterion = SoftmaxCrossEntropyLoss(axis=1, ignore_label=args.ignore_index)
    elif args.bg:
        # ignoring background
        criterion = SoftmaxCrossEntropyLoss(axis=1, ignore_label=0)
    else:
        criterion = SoftmaxCrossEntropyLoss(axis=1)
    if not args.syncbn:
        model.hybridize()
        criterion.hybridize()
    return _init_parallel_model(args, model, criterion)


def _init_parallel_model(args, model, criterion):
    if args.cuda:
        model = ModelDataParallel(model, args.ctx, args.syncbn)
        criterion = CriterionDataParallel(criterion, args.ctx, args.syncbn)
    else:
        model.reset_ctx(ctx=args.ctx)
        criterion.reset_ctx(ctx=args.ctx)
    # resume checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            model.module.collect_params().load(args.resume, ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                .format(args.resume))
    return model, criterion
