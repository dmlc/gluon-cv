"""Base Model for Semantic Segmentation"""
import math
import numpy as np
import mxnet as mx
from mxnet.ndarray import NDArray
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.loss import Loss, _apply_weighting
from ..utils.metrics import voc_segmentation
from ..utils.parallel import parallel_apply
from .resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b
from ..utils.parallel import tuple_map
# pylint: disable=abstract-method,arguments-differ,dangerous-default-value,missing-docstring

__all__ = ['get_segmentation_model', 'SegBaseModel', 'SegEvalModel', 'MultiEvalModel']

def get_segmentation_model(model, **kwargs):
    from .fcn import get_fcn
    from .pspnet import get_psp
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
    }
    return models[model](**kwargs)

class SegBaseModel(HybridBlock):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : Block
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, aux, backbone='resnet50', height=480, width=480, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = resnet50_v1b(pretrained=True, dilated=True, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1b(pretrained=True, dilated=True, **kwargs)
            elif backbone == 'resnet152':
                pretrained = resnet152_v1b(pretrained=True, dilated=True, **kwargs)
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4
        self._up_kwargs = {'height': height, 'width': width}

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c3, c4

    def evaluate(self, x, target=None):
        """evaluating network with inputs and targets"""
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = voc_segmentation.batch_pix_accuracy( \
            pred, target)
        inter, union = voc_segmentation.batch_intersection_union(
            pred, target, self.nclass)

        return correct, labeled, inter, union

    def demo(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class SoftmaxCrossEntropyLoss(Loss):
    """SoftmaxCrossEntropyLoss with ignore labels"""
    def __init__(self, axis=1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, ignore_label=-1, size_average=False, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, axis=self._axis)
        if self._sparse_label:
            if self._size_average:
                valid_label_map = (label != self._ignore_label).astype('float32')
                loss = -(F.pick(pred, label, axis=self._axis, keepdims=True) * valid_label_map)
            else:
                loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
                loss = F.where(label.expand_dims(axis=self._axis) == self._ignore_label,
                               F.zeros_like(loss), loss)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            return F.mean(loss, axis=self._batch_axis, exclude=True) * \
                valid_label_map.size / F.sum(valid_label_map)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)


class SoftmaxCrossEntropyLossWithAux(SoftmaxCrossEntropyLoss):
    """SoftmaxCrossEntropyLoss2D with Auxilary Loss"""
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(SoftmaxCrossEntropyLossWithAux, self).__init__(
            axis=1, ignore_label=ignore_label, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def aux_forward(self, F, pred1, pred2, label, **kwargs):
        loss1 = super(SoftmaxCrossEntropyLossWithAux, self). \
            hybrid_forward(F, pred1, label, **kwargs)
        loss2 = super(SoftmaxCrossEntropyLossWithAux, self). \
            hybrid_forward(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def hybrid_forward(self, F, *inputs, **kwargs):
        if self.aux:
            return self.aux_forward(F, *inputs, **kwargs)
        else:
            return super(SoftmaxCrossEntropyLossWithAux, self). \
                hybrid_forward(F, *inputs, **kwargs)


class SegEvalModel(object):
    """Segmentation Eval Module"""
    def __init__(self, module):
        self.module = module

    def __call__(self, *inputs, **kwargs):
        return self.module.evaluate(*inputs, **kwargs)

    def collect_params(self):
        return self.module.collect_params()


class MultiEvalModel(object):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, ctx_list,
                 base_size=520, crop_size=480, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        self.flip = flip
        self.ctx_list = ctx_list
        self.base_size = base_size
        self.crop_size = crop_size
        self.nclass = nclass
        self.scales = scales
        module.collect_params().reset_ctx(ctx=ctx_list)
        self.evalmodule = SegEvalModel(module)

    def parallel_forward(self, inputs):
        inputs = tuple([tuple([x.as_in_context(ctx)])
                        for (x, ctx) in zip(inputs, self.ctx_list)])
        if len(self.ctx_list) == 1:
            return tuple_map(self(*inputs[0]))
        return parallel_apply(self, inputs, sync=True)

    def __call__(self, image):
        # only single image is supported for evaluation
        image = image.expand_dims(0)
        batch, _, h, w = image.shape
        assert(batch == 1)
        base_size = self.base_size
        crop_size = self.crop_size
        stride_rate = 2.0/3.0
        stride = int(crop_size*stride_rate)
        scores = mx.nd.zeros((batch, self.nclass, h, w), ctx=image.context)
        for scale in self.scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = _resize_image(image, height, width)
            if long_size <= crop_size:
                pad_img = _pad_image(cur_img, crop_size)
                outputs = self.flip_inference(pad_img)
                outputs = _crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = _pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.shape
                assert(ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0*(ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0*(pw-crop_size)/stride)) + 1
                outputs = mx.nd.zeros((batch, self.nclass, ph, pw), ctx=image.context)
                count_norm = mx.nd.zeros((batch, 1, ph, pw), ctx=image.context)
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = _crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = _pad_image(crop_img, crop_size)
                        output = self.flip_inference(pad_crop_img)
                        outputs[:, :, h0:h1, w0:w1] += _crop_image(
                            output, 0, h1-h0, 0, w1-w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert((count_norm == 0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            score = _resize_image(outputs, h, w)
            scores += score

        return scores

    def flip_inference(self, image):
        assert(isinstance(image, NDArray))
        output = self.evalmodule(image)
        if self.flip:
            fimg = _flip_image(image)
            foutput = self.evalmodule(fimg)
            output += _flip_image(foutput)
        return output.exp()

    def collect_params(self):
        return self.evalmodule.collect_params()


def _resize_image(img, h, w):
    return mx.nd.contrib.BilinearResize2D(img, height=h, width=w)


def _pad_image(img, crop_size=480):
    b, c, h, w = img.shape
    assert(c == 3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    mean = [.485, .456, .406]
    std = [.229, .224, .225]
    pad_values = -np.array(mean) / np.array(std)
    img_pad = mx.nd.zeros((b, c, h + padh, w + padw)).as_in_context(img.context)
    for i in range(c):
        img_pad[:, i, :, :] = mx.nd.squeeze(
            mx.nd.pad(img[:, i, :, :].expand_dims(1), 'constant',
                      pad_width=(0, 0, 0, 0, 0, padh, 0, padw),
                      constant_value=pad_values[i]
                     ))
    assert(img_pad.shape[2] >= crop_size and img_pad.shape[3] >= crop_size)
    return img_pad


def _crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def _flip_image(img):
    assert(img.ndim == 4)
    return img.flip(3)
