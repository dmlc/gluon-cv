"""YOLO Demo script."""
import argparse
import mxnet as mx
from mxnet import image
from matplotlib import pyplot as plt
from mxnet.gluon.data.vision import transforms
from .train_cgan import define_G,Resize

def parse_args():
    parser = argparse.ArgumentParser(description='Test with cycle gan networks.')
    parser.add_argument('--images', type=str, required=True,
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='gpu id: e.g. 0. use -1 for CPU')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size, you can increase it when you want to test large image')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
    parser.add_argument('--no_dropout', action='store_false', help='no dropout for the generator')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_args()
    # context list
    if opt.gpu_id == '-1':
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(int(opt.gpu_id.strip()))

    netG = define_G(opt.output_nc, opt.ngf, opt.which_model_netG, not opt.no_dropout)

    # grab some image if not specified
    image_list = [x.strip() for x in opt.images.split(',') if x.strip()]

    netG.load_parameters(opt.pretrained)

    transform_fn = transforms.Compose([
        Resize(opt.loadSize, keep_ratio=False, interpolation=3),
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
        predict = ((predict.transpose([1,2,0]).asnumpy() * 0.5 + 0.5) * 255).clip(0, 255).astype('uint8')
        plt.imshow(predict)
        plt.show()