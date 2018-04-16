import math
import threading
from tqdm import tqdm
import mxnet as mx
from mxnet.ndarray import NDArray

from gluonvision.utils.metrics import voc_segmentation
from gluonvision.model_zoo.segbase import SegEvalModule
from gluonvision.utils.parallel import ModelDataParallel

from utils import *
from option import Options
from utils import get_mask
from data_utils import get_data_loader
from model_utils import get_model_criterion

class MultiEvalModule(object):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, bg, ignore_index=None,
                 base_size=520, crop_size = 480, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        self.bg = bg
        self.flip = flip
        self.base_size = base_size
        self.crop_size = crop_size
        if ignore_index is not None:
            self.nclass = nclass - 1
        else:
            self.nclass = nclass
        self.evalmodule = SegEvalModule(module, bg)
        self.scales=scales

    def __call__(self, image, target=None):
        """
        multi-size evaluation
        image: 4D Variable 1x3xHxW
        target: 3D variable 1xHxW
        """
        # only single image is supported for evaluation
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
                pad_img = pad_image(cur_img, self.args, crop_size)
                outputs = self.model_forward(pad_img)
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
                        pad_crop_img = pad_image(crop_img, self.args, 
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

        # test mode
        if target is None:
            return scores

        correct, labeled = voc_segmentation.batch_pix_accuracy(
            scores, target, self.bg)
        inter, union = voc_segmentation.batch_intersection_union(
            scores, target, self.nclass, self.bg)
        return correct, labeled, inter, union

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
    evaluator = ModelDataParallel(MultiEvalModule(net.module, args.nclass,
                                  args.bg, args.ignore_index), args.ctx)
    args.test_batch_size = args.ngpus
    test_data = get_data_loader(args)

    tbar = tqdm(test_data)
    for i, (data, im_paths) in enumerate(tbar):
        print('data', data)
        print('im_paths', im_paths)
        raise RuntimeError('debug')
        predicts = evaluator(data)
        for predict, im_path in zip(predicts, im_paths):
            predict = F.squeeze(F.argmax(predict, 1)).asnumpy()
            mask = get_mask(predict, args.dataset)
            outname = os.path.splitext(impath)[0] + '.png'
            mask.save(os.path.join(args.outdir, outname))

         
if __name__ == "__main__":
    args = Options().parse()
    print('Testing model: ', args.resume)
    test(args)
