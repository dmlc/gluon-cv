import os
import math
import threading
from tqdm import tqdm
import mxnet as mx
from mxnet.ndarray import NDArray

from gluonvision.utils.metrics import voc_segmentation
from gluonvision.model_zoo.segbase import SegEvalModule
from gluonvision.utils.parallel import ModelDataParallel, parallel_apply

from utils import *
from option import Options
from utils import get_mask
from data_utils import get_data_loader
from model_utils import get_model_criterion

class MultiEvalModule(object):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, bg, ctx_list,
                 base_size=520, crop_size = 480, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        self.bg = bg
        self.flip = flip
        self.ctx_list = ctx_list
        self.base_size = base_size
        self.crop_size = crop_size
        self.nclass = nclass
        self.evalmodule = SegEvalModule(module, bg)
        self.scales=scales

    def parallel_forward(self, inputs):
        inputs = [x.as_in_context(ctx) for (x, ctx) in zip(inputs, self.ctx_list)]
        if len(self.ctx_list) == 1:
            return self(*inputs[0])
        return parallel_apply(self, inputs, sync=True)

    def __call__(self, image, target=None):
        # only single image is supported for evaluation
        image = image.expand_dims(0)
        batch, _, h, w = image.shape
        assert(batch == 1)
        base_size = self.base_size
        crop_size = self.crop_size
        stride_rate = 2.0/3.0
        stride = int(crop_size*stride_rate)
        
        scores = mx.nd.zeros((batch,self.nclass,h,w), ctx=image.context)
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
            cur_img = resize_image(image, height, width)
            if scale <= 1.25 or long_size <= crop_size:# #
                pad_img = pad_image(cur_img, crop_size)
                outputs = self.model_forward(pad_img)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.shape
                assert(ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0*(ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0*(pw-crop_size)/stride)) + 1
                outputs = mx.nd.zeros((batch,self.nclass,ph,pw), ctx=image.context)
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
                        pad_crop_img = pad_image(crop_img, 
                            crop_size)
                        output = self.model_forward(pad_crop_img)
                        outputs[:,:,h0:h1,w0:w1] += crop_image(output,
                            0, h1-h0, 0, w1-w0)
                        count_norm[:,:,h0:h1,w0:w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]

            score = resize_image(outputs, h, w)
            scores += score

        if target is None:
            return scores

    def model_forward(self, image):
        assert(isinstance(image, NDArray))
        output = self.evalmodule(image)
        if self.flip:
            fimg = flip_image(image)
            foutput =self.evalmodule(fimg)
            output += flip_image(foutput)
        return output.exp()

    def collect_params(self):
        return self.evalmodule.collect_params()

def test(args):
    net, criterion = get_model_criterion(args)
    # module, nclass, bg, ctx_list,
    evaluator = MultiEvalModule(net.module, args.nclass,
                                args.bg, args.ctx)
    args.test_batch_size = args.ngpus
    test_data = get_data_loader(args)

    tbar = tqdm(test_data)
    for i, (data, im_paths) in enumerate(tbar):
        predicts = evaluator.parallel_forward(data)
        for predict, impath in zip(predicts, im_paths):
            predict = mx.nd.squeeze(mx.nd.argmax(predict, 1)).asnumpy()
            mask = get_mask(predict, args.dataset)
            outname = os.path.splitext(impath)[0] + '.png'
            mask.save(os.path.join(args.outdir, outname))


if __name__ == "__main__":
    args = Options().parse()
    args.test = True
    print('Testing model: ', args.resume)
    test(args)
