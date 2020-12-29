import argparse
import random
from mxnet import gluon, image,autograd
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
from mxnet.gluon import nn
import mxnet.ndarray as nd
import mxnet as mx
from mxboard import SummaryWriter
import os
from mxnet.gluon.model_zoo import vision

class ResnetBlock(gluon.nn.HybridBlock):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.HybridSequential()
        with self.name_scope():
            self.conv_block.add(
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm()
            )


    def hybrid_forward(self, F, x,*args, **kwargs):
        out = self.conv_block(x)
        return out + x

class SubpixelBlock(gluon.nn.HybridBlock):
    def __init__(self):
        super(SubpixelBlock, self).__init__()
        self.conv = nn.Conv2D(256, kernel_size=3, strides=1,padding=1)
        self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv(x)
        x = x.reshape((0, -4, -1, 4, 0, 0))  # bs, c // 4, h, w
        x = x.reshape((0, 0, -4, 2, 2, 0, 0))  # bs, c // 4, 2, 2, h, w
        x = x.transpose((0, 1, 2, 4, 3, 5))  # bs, c // 4, 2, h, 2, w
        x = x.reshape((0, 0, -3, -3))  # bs, c // 4, h * 2, w * 2
        x = self.relu(x)
        return x

class SRGenerator(gluon.nn.HybridBlock):
    def __init__(self):
        super(SRGenerator, self).__init__()
        self.conv1 = nn.Conv2D(64, kernel_size=3, strides=1,padding=1,activation='relu')
        self.res_block = nn.HybridSequential()
        with self.name_scope():
            for i in range(16):
                self.res_block.add(
                    ResnetBlock()
                )

            self.res_block.add(
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm()
            )
        self.subpix_block1 = SubpixelBlock()
        self.subpix_block2 = SubpixelBlock()
        self.conv4 = nn.Conv2D(3,kernel_size=1,strides=1,activation='tanh')

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.conv1(x)
        out = self.res_block(x)
        x = out + x
        x = self.subpix_block1(x)
        x = self.subpix_block2(x)
        x = self.conv4(x)
        return x

class ConvBlock(gluon.nn.HybridSequential):
    def __init__(self,filter_num,kernel_size=4,stride=2,padding=1):
        super(ConvBlock,self).__init__()
        self.model = nn.HybridSequential()
        with self.name_scope():
            self.model.add(
                nn.Conv2D(filter_num, kernel_size, stride,padding,use_bias=False),
                nn.BatchNorm(),
                nn.LeakyReLU(0.2),
            )

    def hybrid_forward(self, F, x,*args, **kwargs):
        return self.model(x)


class SRDiscriminator(gluon.nn.HybridBlock):
    def __init__(self):
        super(SRDiscriminator,self).__init__()
        self.model = nn.HybridSequential()
        self.res_block = nn.HybridSequential()
        df_dim = 64
        with self.name_scope():
            self.model.add(
                nn.Conv2D(df_dim, 4, 2,1),
                nn.LeakyReLU(0.2)
            )
            for i in [2,4,8,16,32]:
                self.model.add(ConvBlock(df_dim * i ))
            self.model.add(ConvBlock(df_dim * 16,1,1,padding=0))
            self.model.add(
                nn.Conv2D(df_dim * 8, 1, 1,use_bias=False),
                nn.BatchNorm()
            )
            self.res_block.add(
                ConvBlock(df_dim * 2, 1,1),
                ConvBlock(df_dim * 2, 3, 1),
                nn.Conv2D(df_dim * 8, 3, 1,use_bias=False),
                nn.BatchNorm()
            )
        self.lrelu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.dense = nn.Dense(1)

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.model(x)
        #23
        out = self.res_block(x)
        x = out + x
        x = self.lrelu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def weights_init(params):
    for param_name in params:
        param = params[param_name]
        if param_name.find('conv') != -1:
            if param_name.find('weight') != -1:
                param.set_data(nd.random.normal(0.0,0.02,shape=param.data().shape))
            elif param_name.find('bias') != -1:
                param.set_data(nd.zeros(param.data().shape))
        elif param_name.find('batchnorm') != -1:
            if param_name.find('gamma') != -1:
                param.set_data(nd.random.normal(1.0, 0.02,shape=param.data().shape))
            elif param_name.find('beta') != -1:
                param.set_data(nd.zeros(param.data().shape))

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

class DataSet(gluon.data.Dataset):
    def __init__(self,root,crop_transform,downsample_transform,last_transform):
        self.dir = root
        self.paths = [os.path.join(self.dir,f) for f in os.listdir(self.dir)]
        self.paths = sorted(self.paths)
        self.crop_transform = crop_transform
        self.downsample_transform = downsample_transform
        self.last_transform = last_transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = image.imread(path)

        hr_img = self.crop_transform(img)
        lr_img = self.downsample_transform(hr_img)
        hr_img = self.last_transform(hr_img)
        lr_img = self.last_transform(lr_img)
        return hr_img,lr_img

    def __len__(self):
        return len(self.paths)

# def plot_img(losses_log):
def plot_loss(losses_log,global_step,epoch, i):
    message = '(epoch: %d, iters: %d) ' % (epoch, i)
    for key,value in losses_log.losses.items():
        if 'err' in key:
            loss = nd.concatenate(value,axis=0).mean().asscalar()
            sw.add_scalar('err', {key : loss}, global_step)
            message += '%s: %.6f ' % (key, loss)
    print(message)

def plot_img(losses_log):
    sw.add_image(tag='lr_img', image=nd.clip(nd.concatenate(losses_log['lr_img'])[0:4], 0, 1))
    sw.add_image(tag='hr_img', image=nd.clip(nd.concatenate(losses_log['hr_img'])[0:4], 0, 1))
    sw.add_image(tag='hr_img_fake', image=nd.clip(nd.concatenate(losses_log['hr_img_fake'])[0:4], 0, 1))

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

def mse_loss(input,target):
    e = ((input - target) ** 2).mean(axis=0, exclude=True)
    return e

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--experiment', default=None, help='Where to store models')
    parser.add_argument('--fineSize', type=int, default=384, help='then crop to this size')
    parser.add_argument('--n_epoch_init', type=int, default=100, help='# of iter at pretrained')
    parser.add_argument('--n_epoch', type=int, default=20000, help='# of iter at training')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--lr_init', type=float, default=1e-4, help='initial learning rate for pretrain')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='initial learning rate for adam')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_args()

    print(opt)
    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    decay_every = int(opt.n_epoch / 2)

    opt.manualSeed = 10#random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    mx.random.seed(opt.manualSeed)

    if opt.experiment is None:
        opt.experiment = 'samples'
    os.system('mkdir {0}'.format(opt.experiment))

    if opt.gpu_ids == '-1':
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in opt.gpu_ids.split(',') if i.strip()]

    dummy_img = nd.random.uniform(0,1,(1,3,int(opt.fineSize/4),int(opt.fineSize/4)),ctx=mx.gpu(0))
    netG = SRGenerator()
    netD = SRDiscriminator()
    vgg19 = vision.vgg19(pretrained=True,ctx=context)
    features = vgg19.features[:28]

    netG.initialize(mx.initializer.Normal(),ctx=mx.gpu(0))
    dummy_out = netG(dummy_img)
    weights_init(netG.collect_params())
    netG.collect_params().reset_ctx(context)
    netD.initialize(ctx=mx.gpu(0))
    netD(dummy_out)
    weights_init(netD.collect_params())
    netD.collect_params().reset_ctx(context)

    dataset = DataSet(opt.dataroot,RandomCrop(opt.fineSize),transforms.Resize(int(opt.fineSize / 4),interpolation=3),transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers),last_batch='rollover')

    optimizer_G = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': opt.lr_init,'beta1':opt.beta1},kvstore='local')
    optimizer_D = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': opt.lr_init,'beta1':opt.beta1},kvstore='local')

    loss_d = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    for epoch in range(opt.n_epoch_init):
        for i, (hr_img, lr_img) in enumerate(dataloader):
            hr_img_list = gluon.utils.split_and_load(hr_img, ctx_list=context, batch_axis=0)
            lr_img_list = gluon.utils.split_and_load(lr_img, ctx_list=context, batch_axis=0)
            loss_list = []
            with autograd.record():
                for hr_img,lr_img in zip(hr_img_list,lr_img_list):
                    hr_img_predit = netG(lr_img)
                    loss = mse_loss(hr_img_predit,hr_img)
                    loss_list.append(loss)
                autograd.backward(loss_list)
            optimizer_G.step(opt.batchSize)
            print("Epoch %d:  mse: %.8f " % (epoch, nd.concatenate(loss_list).mean().asscalar()))
        netG.save_parameters('{0}/netG_init_epoch_{1}.pth'.format(opt.experiment, epoch))

    real_label = nd.ones((opt.batchSize,))
    fake_label = nd.zeros((opt.batchSize,))
    mean_mask = nd.zeros((opt.batchSize,3,opt.fineSize,opt.fineSize))
    mean_mask[:,0,:,:] = 0.485
    mean_mask[:,1,:,:] = 0.456
    mean_mask[:,2,:,:] = 0.406
    std_mask = nd.zeros((opt.batchSize,3,opt.fineSize,opt.fineSize))
    std_mask[:,0,:,:] = 0.229
    std_mask[:,1,:,:] = 0.224
    std_mask[:,2,:,:] = 0.225
    real_label_list = gluon.utils.split_and_load(real_label, ctx_list=context, batch_axis=0)
    fake_label_list = gluon.utils.split_and_load(fake_label, ctx_list=context, batch_axis=0)
    mean_mask_list = gluon.utils.split_and_load(mean_mask, ctx_list=context, batch_axis=0)
    std_mask_list = gluon.utils.split_and_load(std_mask, ctx_list=context, batch_axis=0)

    # gen_iterations = 0
    losses_log = loss_dict()
    dataloader_len = len(dataloader)
    for epoch in range(0, opt.n_epoch):
        for i, (hr_img, lr_img) in enumerate(dataloader):
            losses_log.reset()
            hr_img_list = gluon.utils.split_and_load(hr_img, ctx_list=context, batch_axis=0)
            lr_img_list = gluon.utils.split_and_load(lr_img, ctx_list=context, batch_axis=0)
            errD_list = []
            hr_img_fake_list = []
            with autograd.record():
                for hr_img, lr_img, real_label,fake_label in zip(hr_img_list,lr_img_list,real_label_list,fake_label_list):
                    output = netD(hr_img).reshape((-1, 1))
                    errD_real = loss_d(output, real_label)

                    hr_img_fake = netG(lr_img)
                    output = netD(hr_img_fake.detach()).reshape((-1, 1))
                    errD_fake = loss_d(output, fake_label)
                    errD = errD_real + errD_fake
                    errD_list.append(errD)
                    hr_img_fake_list.append(hr_img_fake)
                    losses_log.add(errD=errD)
                    losses_log.add(lr_img=lr_img, hr_img=hr_img, hr_img_fake=hr_img_fake)
                autograd.backward(errD_list)
            optimizer_D.step(opt.batchSize)
            errG_list = []
            with autograd.record():
                for hr_img,lr_img,hr_img_fake,mean_mask,std_mask,real_label in zip(hr_img_list,lr_img_list,hr_img_fake_list,mean_mask_list,std_mask_list,real_label_list):
                    errM = mse_loss(hr_img_fake,hr_img)
                    fake_emb = features(((hr_img_fake + 1)/2 - mean_mask)/std_mask)
                    real_emb = features(((hr_img + 1)/2 - mean_mask)/std_mask)
                    errV = 0.006 * mse_loss(fake_emb,real_emb)
                    output = netD(hr_img_fake).reshape((-1, 1))
                    errA = 1e-3 * loss_d(output,real_label)
                    errG = errM + errV + errA
                    errG_list.append(errG)
                    losses_log.add(errG=errG, errM=errM, errV=errV, errA=errA)
                autograd.backward(errG_list)
            optimizer_G.step(opt.batchSize)
            plot_loss(losses_log,epoch * dataloader_len + i,epoch,i)
        plot_img(losses_log)
        if epoch != 0 and (epoch % decay_every == 0):
            optimizer_G.set_learning_rate(optimizer_G.learning_rate * opt.lr_decay)
            optimizer_D.set_learning_rate(optimizer_D.learning_rate * opt.lr_decay)
        if (epoch + 1) % 10 == 0:
            netG.save_parameters('{0}/netG_epoch_{1}.params'.format(opt.experiment, epoch + 1))
            netD.save_parameters('{0}/netD_epoch_{1}.params'.format(opt.experiment, epoch + 1))
