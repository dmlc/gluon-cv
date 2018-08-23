from __future__ import print_function
import argparse
from mxnet.image import image
from mxnet.gluon.data.vision import transforms,CIFAR10,ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon import Trainer,nn
from mxnet.base import numeric_types
from mxnet.initializer import *
from mxnet import autograd
import mxnet as mx
import mxnet.gluon as gluon

import numpy as np
import math
import os
from mxboard import SummaryWriter
from lsun import LSUN

def save_images(images,filename):
    from PIL import Image
    row = int(math.sqrt(len(images)))
    col = row
    height = sum(image.shape[0] for image in images[0:row])
    width = sum(image.shape[1] for image in images[0:col])
    output = np.zeros((height, width, 3))

    for i in range(row):
        for j in range(col):
            image = images[i*row+j]
            h, w, d = image.shape
            output[i*h:i*h + h, j*w:j*w+w] = image
    output = (output * 255).clip(0,255).astype('uint8')
    im = Image.fromarray(output)
    im.save(filename)



class MLP_G(nn.Block):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        with self.name_scope():
            self.main = gluon.nn.Sequential()
            # Z goes into a linear of size: ngf
            self.main.add(nn.Dense(units=ngf, in_units=nz,activation='relu'))
            self.main.add(nn.Dense(units=ngf, in_units=ngf, activation='relu'))
            self.main.add(nn.Dense(units=ngf, in_units=ngf, activation='relu'))
            self.main.add(nn.Dense(units=nc * isize * isize, in_units=ngf))
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.reshape((input.shape[0], input.shape[1]))
        output = self.main(input)
        return output.reshape((output.shape[0], self.nc, self.isize, self.isize))


class MLP_D(nn.Block):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        self.main = gluon.nn.Sequential()
        with self.main.name_scope():
            self.main.add(nn.Dense(units=ndf, in_units=nc * isize * isize, activation='relu'))
            # Z goes into a linear of size: ndf
            self.main.add(nn.Dense(units=ndf, in_units=ndf, activation='relu'))
            self.main.add(nn.Dense(units=ndf, in_units=ndf, activation='relu'))
            self.main.add(nn.Dense(units=1, in_units=ndf))

        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.reshape((input.shape[0],
                           input.shape[1] * input.shape[2] * input.shape[3]))
        output = self.main(input)
        output = output.mean(axis=0)
        return output.reshape(1)


class Resize(nn.Block):
    """Resize an image to the given size.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    keep_ratio : bool
        Whether to resize the short edge or both edges to `size`,
        if size is give as an integer.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.Resize(size=(1000, 500))
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)

    """

    def __init__(self, size, keep_ratio=False, interpolation=1):
        super(Resize, self).__init__()
        self._keep = keep_ratio
        self._size = size
        self._interpolation = interpolation

    def forward(self, x):
        if isinstance(self._size, numeric_types):
            if not self._keep:
                wsize = self._size
                hsize = self._size
            else:
                h, w, _ = x.shape
                if h > w:
                    wsize = self._size
                    hsize = int(h * wsize / w)
                else:
                    hsize = self._size
                    wsize = int(w * hsize / h)
        else:
            wsize, hsize = self._size
        return image.imresize(x, wsize, hsize, self._interpolation)


class DCGAN_D(nn.Block):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        with self.name_scope():
            main = nn.Sequential()
            # input is nc x isize x isize
            main.add(nn.Conv2D(in_channels=nc, channels=ndf, kernel_size=4, strides=2, padding=1, use_bias=False,
                               prefix='initial.conv.{0}-{1}'.format(nc, ndf)))
            main.add(nn.LeakyReLU(0.2, prefix='initial.relu.{0}'.format(ndf)))
            csize, cndf = isize / 2, ndf

            # Extra layers
            for t in range(n_extra_layers):
                main.add(nn.Conv2D(in_channels=cndf, channels=cndf, kernel_size=3, strides=1, padding=1, use_bias=False,
                                   prefix='extra-layers-{0}.{1}.conv'.format(t, cndf)))
                main.add(nn.BatchNorm(in_channels=cndf, prefix='extra-layers-{0}.{1}.batchnorm'.format(t, cndf)))
                main.add(nn.LeakyReLU(0.2, prefix='extra-layers-{0}.{1}.relu'.format(t, cndf)))

            while csize > 4:
                in_feat = cndf
                out_feat = cndf * 2
                main.add(nn.Conv2D(in_channels=in_feat, channels=out_feat, kernel_size=4, strides=2, padding=1,
                                   use_bias=False, prefix='pyramid.{0}-{1}.conv'.format(in_feat, out_feat)))
                main.add(nn.BatchNorm(in_channels=out_feat, prefix='pyramid.{0}.batchnorm'.format(out_feat)))
                main.add(nn.LeakyReLU(0.2, prefix='pyramid.{0}.relu'.format(out_feat)))
                cndf = cndf * 2
                csize = csize / 2

            # state size. K x 4 x 4
            main.add(nn.Conv2D(in_channels=cndf, channels=1, kernel_size=4, strides=1, padding=0, use_bias=False,
                               prefix='final.{0}-{1}.conv'.format(cndf, 1)))
        self.main = main

    def forward(self, input):
        output = self.main(input)

        output = output.mean(axis=0)
        return output.reshape(1)


class DCGAN_G(nn.Block):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        with self.name_scope():
            main = nn.Sequential()
            # input is Z, going into a convolution
            main.add(
                nn.Conv2DTranspose(in_channels=nz, channels=cngf, kernel_size=4, strides=1, padding=0, use_bias=False,
                                   prefix='initial.{0}-{1}.convt'.format(nz, cngf)))
            main.add(nn.BatchNorm(in_channels=cngf, prefix='initial.{0}.batchnorm'.format(cngf)))
            main.add(nn.LeakyReLU(0, prefix='initial.{0}.relu'.format(cngf)))

            csize, cndf = 4, cngf
            while csize < isize // 2:
                main.add(nn.Conv2DTranspose(in_channels=cngf, channels=cngf // 2, kernel_size=4, strides=2, padding=1,
                                            use_bias=False, prefix='pyramid.{0}-{1}.convt'.format(cngf, cngf // 2)))
                main.add(nn.BatchNorm(in_channels=cngf // 2, prefix='pyramid.{0}.batchnorm'.format(cngf // 2)))
                main.add(nn.LeakyReLU(0, prefix='pyramid.{0}.relu'.format(cngf // 2)))
                cngf = cngf // 2
                csize = csize * 2

            # Extra layers
            for t in range(n_extra_layers):
                main.add(nn.Conv2D(in_channels=cngf, channels=cngf, kernel_size=3, strides=1, padding=1, use_bias=False,
                                   prefix='extra-layers-{0}.{1}.conv'.format(t, cngf)))
                main.add(nn.BatchNorm(in_channels=cngf, prefix='extra-layers-{0}.{1}.batchnorm'.format(t, cngf)))
                main.add(nn.LeakyReLU(0, prefix='extra-layers-{0}.{1}.relu'.format(t, cngf)))

            main.add(
                nn.Conv2DTranspose(in_channels=cngf, channels=nc, kernel_size=4, strides=2, padding=1, use_bias=False,
                                   activation='tanh', prefix='final.{0}-{1}.convt'.format(cngf, nc)))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output
    ###############################################################################


class DCGAN_D_nobn(nn.Block):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        with self.name_scope():
            main = nn.Sequential()
            # input is nc x isize x isize
            # input is nc x isize x isize
            main.add(nn.Conv2D(in_channels=nc, channels=ndf, kernel_size=4, strides=2, padding=1, use_bias=False,
                               prefix='initial.conv.{0}-{1}'.format(nc, ndf)))
            main.add(nn.LeakyReLU(0.2, prefix='initial.relu.{0}'.format(ndf)))
            csize, cndf = isize / 2, ndf

            # Extra layers
            for t in range(n_extra_layers):
                main.add(nn.Conv2D(in_channels=cndf, channels=cndf, kernel_size=3, strides=1, padding=1, use_bias=False,
                                   prefix='extra-layers-{0}.{1}.conv'.format(t, cndf)))
                main.add(nn.LeakyReLU(0.2, prefix='extra-layers-{0}.{1}.relu'.format(t, cndf)))

            while csize > 4:
                in_feat = cndf
                out_feat = cndf * 2
                main.add(nn.Conv2D(in_channels=in_feat, channels=out_feat, kernel_size=4, strides=2, padding=1,
                                   use_bias=False, prefix='pyramid.{0}-{1}.conv'.format(in_feat, out_feat)))
                main.add(nn.LeakyReLU(0.2, prefix='pyramid.{0}.relu'.format(out_feat)))
                cndf = cndf * 2
                csize = csize / 2

            # state size. K x 4 x 4
            main.add(nn.Conv2D(in_channels=cndf, channels=1, kernel_size=4, strides=1, padding=0, use_bias=False,
                               prefix='final.{0}-{1}.conv'.format(cndf, 1)))
        self.main = main

    def forward(self, input):
        output = self.main(input)

        output = output.mean(axis=0)
        return output.reshape(1)


class DCGAN_G_nobn(nn.Block):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        with self.name_scope():
            main = nn.Sequential()
            main.add(
                nn.Conv2DTranspose(in_channels=nz, channels=cngf, kernel_size=4, strides=1, padding=0, use_bias=False,
                                   activation='relu', prefix='initial.{0}-{1}.convt'.format(nz, cngf)))

            csize, cndf = 4, cngf
            while csize < isize // 2:
                main.add(nn.Conv2DTranspose(in_channels=cngf, channels=cngf // 2, kernel_size=4, strides=2, padding=1,
                                            use_bias=False, activation='relu',
                                            prefix='pyramid.{0}-{1}.convt'.format(cngf, cngf // 2)))

                cngf = cngf // 2
                csize = csize * 2

            # Extra layers
            for t in range(n_extra_layers):
                main.add(nn.Conv2D(in_channels=cngf, channels=cngf, kernel_size=3, strides=1, padding=1, use_bias=False,
                                   activation='relu', prefix='extra-layers-{0}.{1}.conv'.format(t, cngf)))

            main.add(
                nn.Conv2DTranspose(in_channels=cngf, channels=nc, kernel_size=4, strides=2, padding=1, use_bias=False,
                                   activation='tanh', prefix='final.{0}-{1}.convt'.format(cngf, nc)))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = 10#random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
mx.random.seed(opt.manualSeed)

if opt.cuda:
    context = mx.gpu(0)
else:
    context = mx.cpu()

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = ImageFolderDataset(root=opt.dataroot).transform_first(transforms.Compose([
                                   Resize(opt.imageSize,keep_ratio=True,interpolation=3),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            Resize(opt.imageSize, keep_ratio=True,interpolation=3),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = CIFAR10(root=opt.dataroot,train=True).transform_first(transforms.Compose([
            Resize(opt.imageSize, keep_ratio=True,interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
assert dataset
dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
print('finish init dataloader')
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(layers):
    for layer in layers:
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            layer.weight.set_data(mx.ndarray.random.normal(0.0,0.02,shape=layer.weight.data().shape))
        elif classname.find('BatchNorm') != -1:
            layer.gamma.set_data(mx.ndarray.random.normal(1.0, 0.02,shape=layer.gamma.data().shape))
            layer.beta.set_data(mx.ndarray.zeros(shape=layer.beta.data().shape))

if opt.noBN:
    netG = DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif opt.mlp_G:
    netG = MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    netG = DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
netG.initialize(mx.init.Xavier(factor_type='in',magnitude=0.01),ctx=context)
weights_init(netG.main)
if opt.netG != '': # load checkpoint if needed
    netG.load_params(opt.netG)
print(netG)

if opt.mlp_D:
    netD = MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    netD.initialize(mx.init.Xavier(factor_type='in',magnitude=0.01), ctx=context)
else:
    netD = DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD.initialize(mx.init.Xavier(factor_type='in',magnitude=0.01), ctx=context)
    weights_init(netD.main)
if opt.netD != '':
    netD.load_params(opt.netD)
print(netD)

input = mx.nd.zeros((opt.batchSize, 3, opt.imageSize, opt.imageSize))
noise = mx.nd.zeros((opt.batchSize, nz, 1, 1))
fixed_noise = mx.ndarray.random.normal(shape=(opt.batchSize, nz, 1, 1))
one = mx.nd.array([1])
mone = one * -1

# setup optimizer
if opt.adam:
    trainerD = Trainer(netD.collect_params(),optimizer='adam',optimizer_params={'learning_rate': opt.lrD,'beta1': opt.beta1,'beta2':0.999})
    trainerG = Trainer(netG.collect_params(),optimizer='adam',optimizer_params={'learning_rate': opt.lrG, 'beta1': opt.beta1, 'beta2': 0.999})
else:
    trainerD = Trainer(netD.collect_params(),optimizer='rmsprop',optimizer_params={'learning_rate': opt.lrD,'gamma1':0.99,'gamma2':0.99,'epsilon':1e-12})
    trainerG = Trainer(netG.collect_params(),optimizer='rmsprop', optimizer_params={'learning_rate': opt.lrG,'gamma1':0.99,'gamma2':0.99,'epsilon':1e-14})

print('start training')

sw = SummaryWriter(logdir='./logs', flush_secs=5)

netD.hybridize()
netG.hybridize()

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1
            # clamp parameters to a cube
            for p in netD.collect_params():
                param = netD.collect_params(p)[p]
                param.set_data(mx.nd.clip(param.data(), opt.clamp_lower, opt.clamp_upper))

            data = next(data_iter)[0]
            data = data.as_in_context(context)
            i += 1

            # train with real
            batch_size = data.shape[0]

            with autograd.record():
                errD_real = netD(data)

                # train with fake
                noise = mx.ndarray.random.normal(shape=(opt.batchSize, nz, 1, 1), ctx=context)
                fake = netG(noise)
                errD_fake = netD(fake.detach())
                errD = errD_real - errD_fake
                errD.backward()
            trainerD.step(1)

        ############################
        # (2) Update G network
        ###########################
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise =  mx.ndarray.random.normal(shape=(opt.batchSize, nz, 1, 1), ctx=context)
        with autograd.record():
            fake = netG(noise)
            errG = netD(fake)
            errG.backward()
        trainerG.step(1)
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(dataloader), gen_iterations,
            errD.asnumpy()[0], errG.asnumpy()[0], errD_real.asnumpy()[0], errD_fake.asnumpy()[0]))

        sw.add_scalar(
            tag='loss_D',
            value=-errD.asnumpy()[0],
            global_step=gen_iterations
        )

        if gen_iterations % 500 == 0:
            real_cpu = data * 0.5 + 0.5
            save_images(real_cpu.asnumpy().transpose(0, 2, 3, 1), '{0}/real_samples.png'.format(opt.experiment))
            fake = netG(fixed_noise.as_in_context(context))
            fake = fake * 0.5 + 0.5
            save_images(fake.asnumpy().transpose(0, 2, 3, 1), '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
    netG.save_params('{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    netD.save_params('{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
