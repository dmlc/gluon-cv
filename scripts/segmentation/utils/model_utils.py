import os
import shutil
import importlib

import mxnet as mx
import mxnet.ndarray as F
from mxnet import gluon

from gluonvision.utils.parallel import ModelDataParallel, CriterionDataParallel

__all__ = ['save_checkpoint', 'get_model_criterion']


def save_checkpoint(net, args, is_best=False):
    directory = "runs/%s/%s/%s/" % (args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename='checkpoint.params'
    filename = directory + filename
    net.save_params(filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.params')


def get_model_criterion(args):
    models = importlib.import_module('gluonvision.model_zoo.'+args.model)
    model = models._Net(args.nclass, args.backbone, norm_layer = args.norm_layer, aux = args.aux)
    print('model', model)
    if args.aux:
        Loss = CrossEntropyLossWithAux2d
    else:
        Loss = CrossEntropyLoss2d
    if args.ignore_index is not None:
        # ignoring boundaries
        criterion = Loss(axis=1, ignore_label=args.ignore_index)
    elif args.bg:
        # ignoring background
        criterion = Loss(axis=1, ignore_label=0)
    else:
        criterion = Loss(axis=1)
    if not args.syncbn and not args.test:
        model.hybridize()
        criterion.hybridize()
    return _init_parallel_model(args, model, criterion)


def _init_parallel_model(args, model, criterion):
    model = ModelDataParallel(model, args.ctx, args.syncbn)
    criterion = CriterionDataParallel(criterion, args.ctx, args.syncbn)
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


class CrossEntropyLossWithAux2d(gluon.loss.Loss):
    def __init__(self, axis, aux_weight=0.2, ignore_label=None):
        super(CrossEntropyLossWithAux2d, self).__init__(weight=None, batch_axis=0)
        self._axis = axis
        self.aux_weight = aux_weight
        self._ignore_index = ignore_label
 
    def forward_each(self, F, pred, label):
        pred = F.log_softmax(pred, 1)
        loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        if self._ignore_index is not None:
            mask = (label != self._ignore_index).astype(loss.dtype)
            loss = loss * mask.expand_dims(self._axis)
        loss = F.mean(loss, axis=self._batch_axis, exclude=True)
        if self._ignore_index is not None:
            loss = loss * (1.0 * mask.size / mask.sum())
        return loss

    def hybrid_forward(self, F, pred1, pred2, label):
        loss1 = self.forward_each(F, pred1, label)
        loss2 = self.forward_each(F, pred2, label)
        # Auxilary loss
        return loss1 + self.aux_weight * loss2
