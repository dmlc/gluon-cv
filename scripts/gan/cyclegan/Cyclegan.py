from mxnet import gluon
from mxnet.gluon import data
from mxnet.gluon.data import Dataset
from mxnet import lr_scheduler
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import Block, HybridBlock
from mxboard import SummaryWriter
from mxnet import autograd
from mxnet import gluon
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from PIL import Image
from mxnet.gluon import nn
import os
import argparse
parser = argparse.ArgumentParser(description='Train the CycleGAN model')
parser.add_argument('--usegpu', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--num_epoch', type=int, default=2)
parser.add_argument('--gpu_id', type=int, default=3)
args = parser.parse_args()
if not args.usegpu:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(args.gpu_id)
loss = gluon.loss.L1Loss()


class Gluondataset(Dataset):
    def __init__(self, training=True):
        self.training = training
        if self.training:
            self.train_A_path = sorted(
                [x for x in os.listdir('./datasets/maps/trainA')])
            self.train_B_path = sorted(
                [x for x in os.listdir('./datasets/maps/trainB')])
            for i in range(len(self.train_A_path)):
                self.train_A_path[i] = './datasets/maps/trainA/' + \
                    self.train_A_path[i]
                self.train_B_path[i] = './datasets/maps/trainB/' + \
                    self.train_B_path[i]
        else:
            self.test_A_path = sorted(
                [x for x in os.listdir('./datasets/maps/testA')])
            self.test_B_path = sorted(
                [x for x in os.listdir('./datasets/maps/testB')])
            for i in range(len(self.test_A_path)):
                self.test_A_path[i] = './datasets/maps/testA/' + \
                    self.test_A_path[i]
                self.test_B_path[i] = './datasets/maps/testB/' + \
                    self.test_B_path[i]

    def __getitem__(self, item):
        if self.training:
            train_A_path = self.train_A_path[item]
            train_B_path = self.train_B_path[item]
            train_Image_A = nd.array(np.asarray(
                Image.open(train_A_path).convert('RGB')))
            train_Image_B = nd.array(np.asarray(
                Image.open(train_B_path).convert('RGB')))
            train_A = self.transform()(train_Image_A)
            train_B = self.transform()(train_Image_B)
            return train_A, train_B
        else:
            test_A_path = self.test_A_path[item]
            test_B_path = self.test_B_path[item]
            test_Image_A = nd.array(
                np.asarray(
                    Image.open(test_A_path).convert('RGB')))
            test_Image_B = nd.array(
                np.asarray(
                    Image.open(test_B_path).convert('RGB')))
            test_A = self.transform()(test_Image_A)
            test_B = self.transform()(test_Image_B)
            return test_A, test_B

    def __len__(self):
        if self.training:
            return len(self.train_A_path)
        else:
            return len(self.test_A_path)

    def transform(self):
        transform_list = []
        transform_list.append(transforms.Resize(286, Image.BICUBIC))
        transform_list.append(transforms.RandomResizedCrop(256))

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)


batch_size = 1
train_data = DataLoader(Gluondataset(), batch_size=batch_size, shuffle=True)
test_data = DataLoader(
    Gluondataset(
        training=False),
    batch_size=batch_size,
    shuffle=False)


class ResnetBlock(HybridBlock):
    def __init__(self, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.model = self.get_blocks()

    def get_blocks(self):
        model = nn.HybridSequential()
        with model.name_scope():
            model.add(nn.ReflectionPad2D(1))
            model.add(nn.Conv2D(256, kernel_size=3))
            model.add(nn.InstanceNorm())
            model.add(nn.Activation(activation='relu'))
            model.add(nn.ReflectionPad2D(1))
            model.add(nn.Conv2D(256, kernel_size=3))
            model.add(nn.InstanceNorm())
        return model

    def forward(self, x):
        out = x + self.model(x)
        return out


class ResnetGenerator(HybridBlock):
    def __init__(self, **kwargs):
        super(ResnetGenerator, self).__init__(**kwargs)
        num_blocks = 9
        with self.name_scope():
            self.module = nn.HybridSequential()
            with self.module.name_scope():
                self.module.add(nn.ReflectionPad2D(3))
                self.module.add(nn.Conv2D(64, kernel_size=7, padding=0))
                self.module.add(nn.InstanceNorm())
                self.module.add(nn.Activation(activation='relu'))
                n_downsampling = 2
                for i in range(n_downsampling):
                    mult = 2**i
                    self.module.add(
                        nn.Conv2D(
                            128 * mult,
                            kernel_size=3,
                            strides=2,
                            padding=1))
                    self.module.add(nn.InstanceNorm())
                    self.module.add(nn.Activation(activation='relu'))
                for i in range(num_blocks):
                    self.module.add(ResnetBlock())
                for i in range(n_downsampling):
                    mult = 2**(n_downsampling - i)
                    self.module.add(nn.Conv2DTranspose(
                        int(64 * mult / 2), kernel_size=3, strides=2, padding=1, output_padding=1))
                    self.module.add(nn.InstanceNorm())
                    self.module.add(nn.Activation(activation='relu'))
                self.module.add(nn.ReflectionPad2D(3))
                self.module.add(nn.Conv2D(3, kernel_size=7, padding=0))
                self.module.add(nn.Activation(activation='tanh'))

    def forward(self, x):
        out = self.module(x)
        return out


class NlayerDiscriminator(HybridBlock):
    def __init__(self, **kwargs):
        super(NlayerDiscriminator, self).__init__(**kwargs)
        kw = 4
        padw = 1
        with self.name_scope():
            self.model = nn.HybridSequential()
            with self.model.name_scope():
                self.model.add(
                    nn.Conv2D(
                        64,
                        kernel_size=kw,
                        strides=2,
                        padding=padw))
                self.model.add(nn.LeakyReLU(0.2))
                self.model.add(
                    nn.Conv2D(
                        128,
                        kernel_size=kw,
                        strides=2,
                        padding=padw))
                self.model.add(nn.InstanceNorm())
                self.model.add(nn.LeakyReLU(0.2))
                self.model.add(
                    nn.Conv2D(
                        256,
                        kernel_size=kw,
                        strides=2,
                        padding=padw))
                self.model.add(nn.InstanceNorm())
                self.model.add(nn.LeakyReLU(0.2))
                self.model.add(
                    nn.Conv2D(
                        512,
                        kernel_size=kw,
                        strides=2,
                        padding=padw))
                self.model.add(nn.InstanceNorm())
                self.model.add(nn.LeakyReLU(0.2))
                self.model.add(nn.Activation(activation='sigmoid'))

    def forward(self, input):
        out = self.model(input)
        return out


class CycleGANModel(HybridBlock):
    def __init__(self, **kwargs):
        super(CycleGANModel, self).__init__(**kwargs)
        with self.name_scope():
            self.netG_A = ResnetGenerator()
            self.netG_B = ResnetGenerator()
            self.netD_A = NlayerDiscriminator()
            self.netD_B = NlayerDiscriminator()
            self.GAN_real_Loss = gluon.loss.SigmoidBCELoss()
            self.GAN_fake_Loss = gluon.loss.SigmoidBCELoss()
            self.GAN_G_A_Loss = gluon.loss.SigmoidBCELoss()
            self.GAN_G_B_Loss = gluon.loss.SigmoidBCELoss()
            self.criterionCycle_A = gluon.loss.L1Loss()
            self.criterionCycle_B = gluon.loss.L1Loss()
            self.criterionIdt_A = gluon.loss.L1Loss()
            self.criterionIdt_B = gluon.loss.L1Loss()

    def set_optimizer(self):
        self.trainer_G_A = gluon.Trainer(
            self.netG_A.collect_params(), 'adam', {
                'learning_rate': args.lr, 'beta1': 0.5, 'beta2': 0.999})
        self.trainer_G_B = gluon.Trainer(
            self.netG_B.collect_params(), 'adam', {
                'learning_rate': args.lr, 'beta1': 0.5, 'beta2': 0.999})
        self.trainer_D_A = gluon.Trainer(
            self.netD_A.collect_params(), 'adam', {
                'learning_rate': args.lr, 'beta1': 0.5, 'beta2': 0.999})
        self.trainer_D_B = gluon.Trainer(
            self.netD_B.collect_params(), 'adam', {
                'learning_rate': args.lr, 'beta1': 0.5, 'beta2': 0.999})

    def set_input(self, input):
        self.real_A = input[0]
        self.real_B = input[1]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.GAN_real_Loss(
            pred_real, mx.nd.ones(
                pred_real.shape).as_in_context(ctx))
        pred_fake = netD(fake.detach())
        loss_D_fake = self.GAN_fake_Loss(
            pred_fake, mx.nd.zeros(
                pred_fake.shape).as_in_context(ctx))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self.real_A, self.fake_A)

    def backward_G(self):
        lambda_idt = 0.5
        lambda_A = 10
        lambda_B = 10
        target = mx.nd.ones(self.netD_A(self.fake_B).shape).as_in_context(ctx)
        self.idt_A = self.netG_A(self.real_B)
        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_A = self.criterionIdt_A(
            self.idt_A, self.real_B) * lambda_B * lambda_idt
        self.loss_idt_B = self.criterionIdt_B(
            self.idt_B, self.real_A) * lambda_A * lambda_idt
        self.loss_G_A = self.GAN_G_A_Loss(self.netD_A(self.fake_B), target)
        self.loss_G_B = self.GAN_G_B_Loss(self.netD_B(self.fake_A), target)
        self.loss_cycle_A = self.criterionCycle_A(
            self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle_B(
            self.rec_B, self.real_B) * lambda_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_idt_A + \
            self.loss_idt_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def optimizer_parameters(self):
        with autograd.record():
            self.forward()
            self.backward_G()
        self.trainer_G_A.step(1)
        self.trainer_G_B.step(1)
        with autograd.record():
            self.backward_D_A()
            self.backward_D_B()
        self.trainer_D_A.step(1)
        self.trainer_D_B.step(1)


num_epoch = args.num_epoch
net = CycleGANModel()
net.initialize(mx.init.Xavier(), ctx=ctx)
net.set_optimizer()
net.hybridize()
for epoch in range(num_epoch):
    DA_loss_list = []
    DB_loss_list = []
    GA_loss_list = []
    GB_loss_list = []
    for idx, (train_A, train_B) in enumerate(train_data):
        train_A = train_A.as_in_context(ctx)
        train_B = train_B.as_in_context(ctx)
        net.set_input((train_A, train_B))
        net.optimizer_parameters()
        DA_loss_list.append(net.loss_D_A.asscalar())
        DB_loss_list.append(net.loss_D_B.asscalar())
        GA_loss_list.append(net.loss_G_A.asscalar())
        GB_loss_list.append(net.loss_G_B.asscalar())
    DA_loss = sum(DA_loss_list) / len(DA_loss_list)
    DB_loss = sum(DB_loss_list) / len(DB_loss_list)
    GA_loss = sum(GA_loss_list) / len(GA_loss_list)
    GB_loss = sum(GB_loss_list) / len(GB_loss_list)
    with SummaryWriter(logdir='./logs') as sw:
        sw.add_scalar(tag='DA_Loss', value=DA_loss, global_step=epoch + 1)
        sw.add_scalar(tag='DB_Loss', value=DB_loss, global_step=epoch + 1)
        sw.add_scalar(tag='GA_Loss', value=GA_loss, global_step=epoch + 1)
        sw.add_scalar(tag='GB_Loss', value=GB_loss, global_step=epoch + 1)
    print('Epoch {}, DA_loss: {:.4f}, DB_loss: {:.4f}, GA_loss: {:.4f}, GB_loss: {:.4f}'
          .format(epoch + 1, DA_loss, DB_loss, GA_loss,
                  GB_loss))


test_A = Gluondataset(training=False).__getitem__(0)[0]
test_A = mx.nd.array(test_A.expand_dims(0))
test_A = test_A.as_in_context(mx.cpu())
net.collect_params().reset_ctx(mx.cpu())
data = net.netG_A(test_A)
data = data * 0.5 + 0.5
with SummaryWriter(logdir='./logs') as sw:
    sw.add_image(tag='testdemo', image=data)
