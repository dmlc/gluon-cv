import math
import threading
import mxnet as mx
from mxnet.ndarray import NDArray

import gluonvision.utils as utils
from utils import *

class Evaluator:
    def __init__(self, args):
        self.idx = 0
        self.args = args
        self.dataset = args.dataset
        self.lock = threading.Lock()
        if args.ignore_index is not None:
            self.nclass = args.nclass - 1
        else:
            self.nclass = args.nclass

    def test_batch(self, outputs, targets):
        # for single gpu
        if isinstance(outputs, NDArray):
            correct, labeled = utils.batch_pix_accuracy(outputs,
                targets, self.args.bg)
            inter, union = utils.batch_intersection_union(
                outputs, targets, self.nclass, self.args.bg)
            return correct, labeled

        if len(outputs) == 1:
            return self.test_batch(outputs[0], targets[0])

        corrects, labeleds, inters, unions = {}, {}, {}, {}
        def _worker(i, output, target, lock):
            try:
                correct, labeled = utils.batch_pix_accuracy( \
                   output, target, self.args.bg)
                inter, union = utils.batch_intersection_union(
                    output, target, self.nclass, self.args.bg)
                with lock:
                    corrects[i], labeleds[i] = correct, labeled
                    inters[i], unions[i] = inter, union
            except Exception as e:
                with lock:
                    corrects[i], labeleds[i] = e, e
                    inters[i], unions[i] = e, e 

        # multi-threading for different gpu
        threads = [threading.Thread(target=_worker,
                                    args=(i, output, target, self.lock),
                                    )
                   for i, (output, target) in
                   enumerate(zip(outputs, targets))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # sum for pixAcc
        def _list_gather(x):
            y = []
            for i in range(len(x)):
                xi = x[i]
                if isinstance(xi, Exception):
                    raise xi
                y.append(xi)
            output = mx.nd.array(y)
            return output.sum().asnumpy()[0]

        # sum for IoU
        def _tensor_gather(x):
            y = 0
            for i in range(len(x)):
                xi = x[i]
                if isinstance(xi, Exception):
                    raise xi
                y += xi
            return y

        # gather the output
        correct, labeled = _list_gather(corrects), _list_gather(labeleds)
        inter, union = _tensor_gather(inters),  _tensor_gather(unions)
        return correct, labeled, inter, union

    def multi_eval_batch(self, image, model, target=None, flip=True):
        """
        multi-size evaluation
        image: 4D Variable 1x3xHxW
        target: 3D variable 1xHxW
        """
        # only single image is supported for evaluation
        batch, _, h, w = image.shape
        assert(batch == 1)
        base_size = 520
        crop_size = 480
        stride_rate = 2.0/3.0
        stride = int(crop_size*stride_rate)
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        scores = mx.nd.zeros((batch,self.args.nclass,h,w), ctx=image.context)
        for scale in scales:
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
            cur_img = resize_image(image, height, width)
            if scale <= 1.25 or long_size <= crop_size:# #
                pad_img = pad_image(cur_img, self.args, crop_size)
                outputs = self.model_forward(model, pad_img, flip)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.args, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.shape
                assert(ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0*(ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0*(pw-crop_size)/stride)) + 1
                outputs = mx.nd.zeros((batch,self.args.nclass,ph,pw), ctx=image.context)
                count_norm = mx.nd.zeros((batch,1,ph,pw), ctx=image.context)
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = pad_image(crop_img, self.args, 
                            crop_size)
                        output = self.model_forward(model, pad_crop_img, 
                            flip)
                        outputs[:,:,h0:h1,w0:w1] += crop_image(output,
                            0, h1-h0, 0, w1-w0)
                        count_norm[:,:,h0:h1,w0:w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]

            score = resize_image(outputs, h, w)
            scores += score

        # test mode
        if target is None:
            return scores
        correct, labeled = utils.batch_pix_accuracy(
            scores, target, self.args.bg)
        inter, union = utils.batch_intersection_union(
            scores, target, self.nclass, self.args.bg)
        return correct, labeled, inter, union

    def model_forward(self, model, image, flip=True):
        assert(isinstance(image, NDArray))
        output = model(image)
        if self.args.aux:
            output = output[0]
        if flip:
            fimg = flip_image(image)
            foutput = model(fimg)
            if self.args.aux:
                output += flip_image(foutput[0])
            else:
                output += flip_image(foutput)
        return output.exp()
