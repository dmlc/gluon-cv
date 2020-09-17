import os
import random
import math
import argparse
import logging
import os.path as osp
from io import BytesIO
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader

from model import StyledGenerator, Discriminator
# pylint: disable-all

class MultiResolutionDataset(gluon.data.Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = np.asarray(Image.open(buffer))
        img = self.transform(nd.array(img))

        return img


def requires_grad(model, flag=True):
    mx_params = model.collect_params()
    for p in mx_params:
        if flag:
            mx_params[p].grad_req = 'write'
        else:
            mx_params[p].grad_req = 'null'


def accumulate(model1, model2, decay=0.999):
    par1 = model1.collect_params()
    par2 = model2.collect_params()
    par1.reset_ctx(mx.cpu())
    par2.reset_ctx(mx.cpu())

    requires_grad(model1, False)
    key_dict = {'hybridsequential0':'hybridsequential5', 
               'hybridsequential1':'hybridsequential6', 
               'hybridsequential2':'hybridsequential7'}
    
    for k in par2.keys():
        k2 = k.split('_')[0]
        k1 = k.replace(k2, key_dict[k2], 1)
        par1[k1].set_data(par1[k1].data()*decay+((1-decay)*par2[k].data()))

    par1.reset_ctx(mx.gpu())
    par2.reset_ctx(context)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, last_batch='discard')
    return loader


def adjust_lr(optimizer, lr):
    optimizer.set_learning_rate(lr)


def normalize_image(img, dmin, dmax):
    result = img.copy()
    result = nd.clip(img, dmin, dmax)
    result = (result - dmin)/(dmax - dmin + 1e-5)
    return result


def plot_images(images, path, ncols, nrows):
    fig = plt.figure(figsize=(25, 25))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, img in zip(grid, images):
        norm_img = normalize_image(img, -1, 1)
        img = nd.clip(norm_img * 255 + 0.5, 0, 255).asnumpy().astype(np.uint8) 
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')

    plt.savefig(path, bbox_inches='tight')


def train(args, dataset, generator, discriminator):


    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, args.lr_default))
    adjust_lr(d_optimizer, args.lr.get(resolution, args.lr_default))

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    requires_grad(generator, False)
    requires_grad(discriminator, True)


    pbar = tqdm(range(200_000))

    for i in pbar:

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            generator.save_parameters(osp.join(args.ckpt_dir, f'generator_step-{ckpt_step}.params'))
            discriminator.save_parameters(osp.join(args.ckpt_dir, f'discriminator_step-{ckpt_step}.params'))
            g_running.save_parameters(osp.join(args.ckpt_dir, f'g_running_step-{ckpt_step}.params'))

            adjust_lr(g_optimizer, args.lr.get(resolution, args.lr_default))
            adjust_lr(d_optimizer, args.lr.get(resolution, args.lr_default))

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]
        b_size = real_image.shape[0]
        real_image_list = gluon.utils.split_and_load(real_image, ctx_list=context, batch_axis=0)

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = nd.random.randn(
                4, b_size, code_size).split(4, 0)
            gen_in1 = [nd.squeeze(gen_in11, axis=0), nd.squeeze(gen_in12, axis=0)]
            gen_in2 = [nd.squeeze(gen_in21, axis=0), nd.squeeze(gen_in22, axis=0)]

        else:
            gen_in1, gen_in2 = nd.random.randn(2, b_size, code_size).split(2, 0)
            gen_in1 = nd.squeeze(gen_in1, axis=0)
            gen_in2 = nd.squeeze(gen_in2, axis=0)

        gen_in1_list = gluon.utils.split_and_load(gen_in1, ctx_list=context, batch_axis=0)
        gen_in2_list = gluon.utils.split_and_load(gen_in2, ctx_list=context, batch_axis=0)

        if args.loss == 'wgan':
            fake_predict_list = []
            real_predict_list = []
            D_loss_list = []
            with autograd.record():
                for _, (rl_image, g1) in enumerate(zip(real_image_list, gen_in1_list)):      
                    real_predict = discriminator(rl_image, step, alpha)
                    real_predict = -real_predict.mean()
                    real_predict_list.append(real_predict)

                    fake_image = generator(g1, step, alpha)
                    fake_predict = discriminator(fake_image.detach(), step, alpha)
                    fake_predict = fake_predict.mean()
                    fake_predict_list.append(fake_predict)

                    D_loss_list.append(real_predict+fake_predict)

            autograd.backward(loss_list)

        elif args.loss == 'r1':
            # Not able to implement r1 loss
            raise Exception('r1 loss has not been implemented, please use wgan loss')
        else:
            raise Exception('Not valid loss, please use wgan loss')

        if i%10 == 0:
            real_predict_val = [i.asnumpy() for i in real_predict_list]
            fake_predict_val = [i.asnumpy() for i in fake_predict_list]
            d_real_val = np.concatenate(real_predict_val).mean()
            d_fake_val = np.concatenate(fake_predict_val).mean()
            disc_loss_val = d_real_val + d_fake_val

        d_optimizer.step(b_size, ignore_stale_grad=True)

        if (i +1) % n_critic == 0:

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            if args.loss == 'wgan-gp':
                predict_list = []
                with autograd.record():
                    for _, g2 in enumerate(gen_in2_list):
                        fake_image = generator(g2, step, alpha)
                        predict = discriminator(fake_image, step, alpha)
                        predict = -predict.mean()
                        predict_list.append(predict)
                autograd.backward(predict_list)
            elif args.loss == 'r1':
                # Not able to implement r1 loss
                raise Exception('r1 loss has not been implemented, please use wgan loss')
            else:
                raise Exception('Not valid loss, please use wgan loss')

            if i%10 == 0:
                predict_val = [i.asnumpy() for i in predict_list]
                gen_loss_val = np.concatenate(predict_val).mean()

            g_optimizer.step(b_size, ignore_stale_grad=True)

            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i+1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            for _ in range(gen_i):
                results = g_running(
                        nd.random.randn(gen_j, code_size, ctx=mx.gpu(0)), step, alpha
                    )
                for r in results:
                    images.append(r)

            plot_images(images, osp.join(args.out, f'{str(i + 1).zfill(6)}.png'), gen_i, gen_j)

        if (i+1) % 1000 == 0:
            generator.save_parameters(osp.join(args.ckpt_dir, f'g-{str(i + 1).zfill(6)}.params'))
            discriminator.save_parameters(osp.join(args.ckpt_dir, f'd-{str(i + 1).zfill(6)}.params'))
            g_running.save_parameters(osp.join(args.ckpt_dir, f'g_running-{str(i + 1).zfill(6)}.params'))

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.1f}; D: {disc_loss_val:.1f};'
            f'D_real: {d_real_val:.1f}; D_fake: {d_fake_val:.1f}; Alpha: {alpha:.4f}'
        )

        logger.info(f'Size: {4 * 2 ** step}; G: {gen_loss_val:.1f}; D: {disc_loss_val:.1f}\
            D_real: {d_real_val:1f}; D_fake: {d_fake_val:1f}; Alpha: {alpha:.4f}')

        pbar.set_description(state_msg)


if __name__ == '__main__':

    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('--path', type=str, help='path of specified dataset')
    parser.add_argument('--phase', type=int, default=500_000, help='number of samples used for each training phases')
    parser.add_argument('--lr_default', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument('--ckpt_g', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--ckpt_d', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--ckpt_g_running', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--no_from_rgb_activate', action='store_true', help='use activate in from_rgb (original implementation)')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('--loss', type=str, default='wgan-gp', choices=['wgan-gp', 'r1'], help='class of gan loss')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--out', type=str, default='sample', help='output directory for saving samples')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoint', help='output directory for saving checkpoints')

    args = parser.parse_args()

    # build logger
    filehandler = logging.FileHandler('stylegan.log')
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(args)

    if args.gpu_ids == '-1':
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in args.gpu_ids.split(',') if i.strip()]

    generator = StyledGenerator(code_size)
    generator.initialize(ctx=context)
    generator.collect_params().reset_ctx(context)

    g_optimizer = gluon.Trainer(generator.collect_params(), optimizer='adam', 
                                optimizer_params={'learning_rate': args.lr_default, 'beta1':0.0, 'beta2':0.99},
                                kvstore='local')

    # Set a different learning rate for style by setting the lr_mult of 0.01
    for k in generator.collect_params().keys():
        if k.startswith('hybridsequential2'):
            generator.collect_params()[k].lr_mult = 0.01


    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    discriminator.initialize(ctx=context)
    discriminator.collect_params().reset_ctx(context)

    d_optimizer = gluon.Trainer(discriminator.collect_params(), optimizer='adam', 
                            optimizer_params={'learning_rate': args.lr_default, 'beta1':0.0, 'beta2':0.99}, kvstore='local')

    g_running = StyledGenerator(code_size)
    g_running.initialize(ctx=mx.gpu(0))
    g_running.collect_params().reset_ctx(mx.gpu(0))
    requires_grad(g_running, False)

    if args.ckpt_g:
        g_running.load_params(args.ckpt_g_running, ctx=mx.gpu(), allow_missing=True)
        generator.load_parameters(args.ckpt_g, ctx=context, allow_missing=True)
        discriminator.load_parameters(args.ckpt_d, ctx=context, allow_missing=True)


    accumulate(g_running, generator, 0)

    transform = transforms.Compose(
        [
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    args.batch_default = 32

    train(args, dataset, generator, discriminator)


