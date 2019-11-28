from train_srgan import SRGenerator
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from matplotlib import pyplot as plt
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
from mxnet import image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test with srgan gan networks.')
    parser.add_argument('--images', type=str, required=True,
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='gpu id: e.g. 0. use -1 for CPU')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_args()
    # context list
    if opt.gpu_id == '-1':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(int(opt.gpu_id.strip()))

    netG = SRGenerator()
    netG.load_parameters(opt.pretrained)
    netG.collect_params().reset_ctx(ctx)
    image_list = [x.strip() for x in opt.images.split(',') if x.strip()]
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ax = None
    for image_path in image_list:
        img = image.imread(image_path)
        img = transform_fn(img)
        img = img.expand_dims(0).as_in_context(ctx)
        output = netG(img)
        predict = mx.nd.squeeze(output)
        predict = ((predict.transpose([1,2,0]).asnumpy() * 0.5 + 0.5) * 255).astype('uint8')
        plt.imshow(predict)
        plt.show()
