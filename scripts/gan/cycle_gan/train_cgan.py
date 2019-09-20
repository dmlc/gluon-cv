import argparse
import random
from mxnet import gluon, image,autograd
from mxnet.gluon.data.vision import transforms
from mxnet.base import numeric_types
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import utils as gutils
from mxnet.gluon.data import DataLoader
from mxnet.gluon import nn
import mxnet.ndarray as nd
import mxnet as mx
import os
import numpy as np
from mxboard import SummaryWriter


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

class RandomCrop(nn.Block):
    def __init__(self, size):
        super(RandomCrop,self).__init__()
        self._size = size

    def forward(self,x):
        h, w, _ = x.shape
        th, tw = (self._size,self._size)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        out = x[i:i + th, j:j + tw, :]
        return out

class ResnetGenerator(gluon.nn.HybridBlock):
    def __init__(self, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.output_nc = output_nc
        self.ngf = ngf
        self.model = nn.HybridSequential()
        with self.name_scope():
            self.model.add(
                nn.ReflectionPad2D(3),
                nn.Conv2D(ngf, kernel_size=7, padding=0),
                nn.InstanceNorm(),
                nn.Activation('relu')
            )

            n_downsampling = 2
            for i in range(n_downsampling):
                mult = 2**i
                self.model.add(
                    nn.Conv2D(ngf * mult * 2, kernel_size=3,strides=2, padding=1),
                    nn.InstanceNorm(),
                    nn.Activation('relu')
                )

            mult = 2**n_downsampling
            for i in range(n_blocks):
                self.model.add(
                    ResnetBlock(ngf * mult, padding_type=padding_type, use_dropout=use_dropout)
                )

            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                self.model.add(
                    nn.Conv2DTranspose(int(ngf * mult / 2),kernel_size=3,strides=2,padding=1,output_padding=1),
                    nn.InstanceNorm(),
                    nn.Activation('relu')
                )
            self.model.add(
                nn.ReflectionPad2D(3),
                nn.Conv2D(output_nc,kernel_size=7,padding=0),
                nn.Activation('tanh')
            )

    def hybrid_forward(self, F, x,*args, **kwargs):
        return self.model(x)

# Define a resnet block
class ResnetBlock(gluon.nn.HybridBlock):
    def __init__(self, dim, padding_type, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout)

    def build_conv_block(self, dim, padding_type, use_dropout):
        conv_block = nn.HybridSequential()
        p = 0
        with self.name_scope():
            if padding_type == 'reflect':
                conv_block.add(nn.ReflectionPad2D(1))
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)

            conv_block.add(
                nn.Conv2D(dim, kernel_size=3, padding=p),
                nn.InstanceNorm(),
                nn.Activation('relu')
            )
            if use_dropout:
                conv_block.add(nn.Dropout(0.5))

            p = 0
            if padding_type == 'reflect':
                conv_block.add(nn.ReflectionPad2D(1))
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            conv_block.add(
                nn.Conv2D(dim, kernel_size=3, padding=p),
                nn.InstanceNorm()
            )

        return conv_block

    def hybrid_forward(self, F, x,*args, **kwargs):
        out = self.conv_block(x)
        return out + x

class UnetGenerator(gluon.nn.HybridBlock):
    def __init__(self, output_nc, num_downs, ngf=64,use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, submodule=unet_block, outermost=True)

        self.model = unet_block

    def hybrid_forward(self, F, x,*args, **kwargs):
        return self.model(x)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(gluon.nn.HybridBlock):
    def __init__(self, outer_nc, inner_nc,submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        downconv = nn.Conv2D(inner_nc, kernel_size=4,strides=2, padding=1)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = nn.InstanceNorm()
        uprelu = nn.Activation('relu')
        upnorm = nn.InstanceNorm()
        self.model = nn.HybridSequential()
        with self.model.name_scope():
            if outermost:
                self.model.add(
                    downconv
                )
                if submodule is not None:
                    self.model.add(
                        submodule
                    )
                self.model.add(
                    uprelu,
                    nn.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding=1),
                    nn.Activation('tanh')
                )
            elif innermost:
                self.model.add(
                    downrelu,
                    downconv,
                    uprelu,
                    nn.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding=1),
                    upnorm
                )
            else:
                self.model.add(
                    downrelu,
                    downconv,
                    downnorm,
                )
                if submodule is not None:
                    self.model.add(
                        submodule
                    )
                self.model.add(
                    uprelu,
                    nn.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding=1),
                    upnorm,
                )
                if use_dropout:
                    self.model.add(nn.Dropout(0.5))

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.outermost:
            return self.model(x)
        else:
            return nd.concat([x, self.model(x)],1)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(gluon.nn.HybridBlock):
    def __init__(self, ndf=64, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.model = nn.HybridSequential()
        kw = 4
        padw = 1
        with self.name_scope():
            self.model.add(
                nn.Conv2D(ndf, kernel_size=kw, strides=2, padding=padw),
                nn.LeakyReLU(0.2),
            )

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult = min(2**n, 8)
                self.model.add(
                    nn.Conv2D(ndf * nf_mult,kernel_size=kw, strides=2, padding=padw),
                    nn.InstanceNorm(),
                    nn.LeakyReLU(0.2),
                )

            nf_mult = min(2**n_layers, 8)
            self.model.add(
                nn.Conv2D(ndf * nf_mult,kernel_size=kw, strides=1, padding=padw),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
            )
            self.model.add(
                nn.Conv2D(1, kernel_size=kw, strides=1, padding=padw)
            )
            if use_sigmoid:
                self.model.add(nn.Activation('sigmoid'))

    def forward(self, x,*args, **kwargs):
        return self.model(x)


def define_G(output_nc, ngf, which_model_netG, use_dropout=False):

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(output_nc, ngf, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(output_nc, ngf, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(output_nc, 7, ngf, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(output_nc, 8, ngf, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % opt.which_model_netG)

    return netG

def define_D(ndf, which_model_netD, n_layers_D=3, use_sigmoid=False):

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(ndf, n_layers=3, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(ndf, n_layers_D, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

    return netD

def weights_init(layers):
    for layer in layers:
        classname = layer.__class__.__name__
        if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            layer.weight.set_data(nd.random.normal(0.0,0.02,shape=layer.weight.data().shape))
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.set_data(nd.zeros(layer.bias.data().shape))
        elif classname.find('BatchNorm') != -1:
            layer.gamma.set_data(nd.random.normal(1.0, 0.02,shape=layer.gamma.data().shape))
            layer.beta.set_data(nd.zeros(layer.bias.data().shape))


class DataSet(gluon.data.Dataset):
    def __init__(self,root,phase,transform):
        self.dir = os.path.join(root,phase)
        self.A_paths = [os.path.join(self.dir+'A',f) for f in os.listdir(self.dir+'A')]
        self.B_paths = [os.path.join(self.dir+'B',f) for f in os.listdir(self.dir+'B')]
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = transform

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = image.imread(A_path)
        B_img = image.imread(B_path)

        A = self.transform(A_img)
        B = self.transform(B_img)
        return A, B

    def __len__(self):
        return max(self.A_size, self.B_size)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        image_array = return_images[0].copyto(images.context)
        for image in return_images[1:]:
            image_array = nd.concat(image_array,image.copyto(images.context),dim=0)
        return image_array

def plot_loss(losses_log,global_step,epoch, i):
    message = '(epoch: %d, iters: %d) ' % (epoch, i)
    for key,value in losses_log.losses.items():
        if 'loss_' in key:
            loss = nd.concatenate(value,axis=0).mean().asscalar()
            sw.add_scalar('loss', {key : loss}, global_step)
            message += '%s: %.3f ' % (key, loss)
    print(message)


def plot_img(losses_log):
    sw.add_image(tag='A', image=nd.clip(nd.concatenate([losses_log['real_A'][0][0:1],
                                                        losses_log['fake_B'][0][0:1],
                                                        losses_log['rec_A'][0][0:1],
                                                        losses_log['idt_A'][0][0:1]]) * 0.5 + 0.5, 0, 1))
    sw.add_image(tag='B', image=nd.clip(nd.concatenate([losses_log['real_B'][0][0:1],
                                                        losses_log['fake_A'][0][0:1],
                                                        losses_log['rec_B'][0][0:1],
                                                        losses_log['idt_B'][0][0:1]]) * 0.5 + 0.5, 0, 1))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True,
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
    parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--no_dropout', action='store_false', help='no dropout for the generator')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_idt', type=float, default=0.5,
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--pool_size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--experiment', default=None, help='Where to store models')
    parser.add_argument('--seed', type=int, default=233, help='Random seed to be fixed.')

    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == '__main__':
    opt = parse_args()

    gutils.random.seed(opt.seed)#random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.seed)

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    if opt.gpu_ids == '-1':
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in opt.gpu_ids.split(',') if i.strip()]

    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    dummy_img = nd.random.uniform(0,1,(1,3,opt.fineSize,opt.fineSize),ctx=mx.gpu(0))
    netG_A = define_G(opt.output_nc,opt.ngf, opt.which_model_netG, not opt.no_dropout)
    netG_B = define_G(opt.output_nc, opt.ngf, opt.which_model_netG, not opt.no_dropout)

    netD_A = define_D(opt.ndf, opt.which_model_netD,opt.n_layers_D, False)
    netD_B = define_D(opt.ndf, opt.which_model_netD,opt.n_layers_D, False)

    nets = [netG_A,netG_B,netD_A,netD_B]
    for net in nets:
        net.initialize(ctx=mx.gpu(0))
        net(dummy_img)
        weights_init(net.model)
        net.collect_params().reset_ctx(context)

    dataset = DataSet(opt.dataroot,'train',transforms.Compose([
                                Resize(opt.loadSize, keep_ratio=True,interpolation=3),
                                RandomCrop(opt.fineSize),
                                transforms.RandomFlipLeftRight(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers),last_batch='rollover')

    fake_A_pool = ImagePool(opt.pool_size)
    fake_B_pool = ImagePool(opt.pool_size)


    optimizer_GA = gluon.Trainer(netG_A.collect_params(), 'adam', {'learning_rate': opt.lr,'beta1':opt.beta1},kvstore='local')
    optimizer_GB = gluon.Trainer(netG_B.collect_params(), 'adam', {'learning_rate': opt.lr,'beta1':opt.beta1},kvstore='local')
    optimizer_DA = gluon.Trainer(netD_A.collect_params(), 'adam', {'learning_rate': opt.lr,'beta1':opt.beta1},kvstore='local')
    optimizer_DB = gluon.Trainer(netD_B.collect_params(), 'adam', {'learning_rate': opt.lr,'beta1':opt.beta1},kvstore='local')
    cyc_loss = gluon.loss.L1Loss()
    def gan_loss(input,target_is_real):
        if target_is_real:
            target = nd.ones(input.shape,ctx=input.context)
        else:
            target = nd.zeros(input.shape, ctx=input.context)
        #mse loss for lsgan
        e = ((input - target) ** 2).mean(axis=0, exclude=True)
        return e

    class loss_dict:
        def __init__(self):
            self.losses = {}

        def add(self, **kwargs):
            for key, value in kwargs.items():
                if key not in self.losses:
                    self.losses[key] = [value]
                else:
                    self.losses[key].append(value)

        def reset(self):
            self.losses = {}

        def __getitem__(self, item):
            return self.losses[item]

    losses_log = loss_dict()
    dataset_size = len(dataloader)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = gluon.utils.split_and_load(real_A, ctx_list=context, batch_axis=0)
            real_B = gluon.utils.split_and_load(real_B, ctx_list=context, batch_axis=0)
            loss_G_list = []
            loss_D_A_list = []
            loss_D_B_list = []
            fake_A_list = []
            fake_B_list = []
            losses_log.reset()
            with autograd.record():
                for A,B in zip(real_A,real_B):
                    fake_B = netG_A(A)
                    rec_A = netG_B(fake_B)
                    fake_A = netG_B(B)
                    rec_B = netG_A(fake_A)

                    # Identity loss
                    idt_A = netG_A(B)
                    loss_idt_A = cyc_loss(idt_A,B) * opt.lambda_B * opt.lambda_idt
                    idt_B = netG_B(A)
                    loss_idt_B = cyc_loss(idt_B,A) * opt.lambda_A * opt.lambda_idt

                    loss_G_A = gan_loss(netD_A(fake_B),True)
                    loss_G_B = gan_loss(netD_B(fake_A),True)
                    loss_cycle_A = cyc_loss(rec_A,A) * opt.lambda_A
                    loss_cycle_B = cyc_loss(rec_B,B) * opt.lambda_B
                    loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

                    loss_G_list.append(loss_G)
                    fake_A_list.append(fake_A)
                    fake_B_list.append(fake_B)
                    losses_log.add(loss_G_A=loss_G_A, loss_cycle_A=loss_cycle_A, loss_idt_A=loss_idt_A,loss_G_B=loss_G_B,
                                   loss_cycle_B=loss_cycle_B, loss_idt_B=loss_idt_B,real_A=A, fake_B=fake_B, rec_A=rec_A,
                                   idt_A=idt_A, real_B=B, fake_A=fake_A, rec_B=rec_B,idt_B=idt_B)
                autograd.backward(loss_G_list)
            optimizer_GA.step(opt.batchSize)
            optimizer_GB.step(opt.batchSize)
            with autograd.record():
                for A,B,fake_A,fake_B in zip(real_A,real_B,fake_A_list,fake_B_list):
                    #train D_A
                    #real
                    fake_B_tmp = fake_B_pool.query(fake_B)
                    pred_real = netD_A(B)
                    loss_D_real = gan_loss(pred_real,True)
                    pred_fake = netD_A(fake_B_tmp.detach())
                    loss_D_fake = gan_loss(pred_fake, False)
                    loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                    loss_D_A_list.append(loss_D_A)

                    #train D_B
                    fake_A_tmp = fake_A_pool.query(fake_A)
                    pred_real = netD_B(A)
                    loss_D_real = gan_loss(pred_real, True)
                    pred_fake = netD_B(fake_A_tmp.detach())
                    loss_D_fake = gan_loss(pred_fake,False)
                    loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                    loss_D_B_list.append(loss_D_B)
                    losses_log.add(loss_D_A=loss_D_A,loss_D_B=loss_D_B)
                autograd.backward(loss_D_A_list + loss_D_B_list)
            optimizer_DA.step(opt.batchSize)
            optimizer_DB.step(opt.batchSize)
            if ((epoch-1) * dataset_size + i) % 100 == 0:
                plot_loss(losses_log, (epoch-1) * dataset_size + i,epoch,i)
                plot_img(losses_log)
        lr = (1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)) * opt.lr
        optimizer_GA.set_learning_rate(lr)
        optimizer_GB.set_learning_rate(lr)
        optimizer_DA.set_learning_rate(lr)
        optimizer_DB.set_learning_rate(lr)
        netG_A.save_parameters('{0}/netG_A_epoch_{1}.params'.format(opt.experiment, epoch))
        netG_B.save_parameters('{0}/netG_B_epoch_{1}.params'.format(opt.experiment, epoch))
        netD_A.save_parameters('{0}/netD_A_epoch_{1}.params'.format(opt.experiment, epoch))
        netD_B.save_parameters('{0}/netD_B_epoch_{1}.params'.format(opt.experiment, epoch))
