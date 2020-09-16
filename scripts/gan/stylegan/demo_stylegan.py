import argparse
import math
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import mxnet as mx
import mxnet.ndarray as nd

from model import StyledGenerator
# pylint: disable-all

def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(nd.random.randn(1024, 512, ctx=device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


def sample(generator, step, mean_style, n_sample, device):

    image = generator(
        nd.random.randn(n_sample, 512, ctx=device), step, 1, mean_style, 0.7,
    )

    return image


def normalize_image(img, dmin, dmax):
    result = img.copy()
    result = nd.clip(img, dmin, dmax)
    result = (result - dmin)/(dmax - dmin + 1e-5)

    return result


def save_image(data, file, normalize=True, img_range=None):

    if img_range is None:
        img_range = [min(data), max(data)]

    norm_img = normalize_image(data, img_range[0], img_range[1])
    img = nd.clip(norm_img * 255 + 0.5, 0, 255).asnumpy().astype(np.uint8) 

    img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    img.save(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_sample', type=int, default=10, help='number of rows of sample matrix')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id: e.g. 0. use -1 for CPU')
    parser.add_argument('--out_dir', type=str, default='samples/', help='output directory for samples')
    parser.add_argument('--path', type=str, default='./stylegan-ffhq-1024px-new.params', 
                        help='path to checkpoint file')
    
    args = parser.parse_args()   

    if args.gpu_id == '-1':
        device = mx.cpu()
    else:
        device = mx.gpu(int(args.gpu_id.strip()))

    generator = StyledGenerator(512, blur=True)

    generator.initialize()
    generator.collect_params().reset_ctx(device)
    generator.load_parameters(args.path, ctx=device)

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2

    imgs = sample(generator, step, mean_style, args.n_sample, device)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir) 

    for i in range(args.n_sample):
        save_image(imgs[i], os.path.join(args.out_dir, 'sample_{}.png'.format(i)), normalize=True, img_range=(-1, 1))
    

