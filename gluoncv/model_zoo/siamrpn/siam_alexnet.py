# coding: utf-8
from __future__ import division
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

class AlexNetLegacy(HybridBlock):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self,width_mult=1,**kwargs):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNet.configs))
        super(AlexNetLegacy, self).__init__()
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(configs[1], kernel_size=11, strides=2),
                    nn.BatchNorm(),
                    nn.MaxPool2D(pool_size=3, strides=2),
                    nn.Activation('relu'))

                self.features.add(nn.Conv2D(configs[2], kernel_size=5),
                    nn.BatchNorm(),
                    nn.MaxPool2D(pool_size=3, strides=2),
                    nn.Activation('relu'))

                self.features.add(nn.Conv2D(configs[3], kernel_size=3),
                    nn.BatchNorm(),
                    nn.Activation('relu'))

                self.features.add(nn.Conv2D(configs[4], kernel_size=3),
                    nn.BatchNorm(),
                    nn.Activation('relu'))

                self.features.add(nn.Conv2D(configs[5], kernel_size=3),
                    nn.BatchNorm())        


    def hybrid_forward(self,F,x):
        x = self.features(x)
        return x

class AlexNet(HybridBlock):
    configs = [3, 96, 256, 384, 384, 256]
    def __init__(self,width_mult=1,**kwargs):
        configs = list(map(lambda x: 3 if x == 3 else
                    int(x*width_mult), AlexNet.configs))
        super(AlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.layer1 = nn.HybridSequential(prefix='')
            self.layer2 = nn.HybridSequential(prefix='')
            self.layer3 = nn.HybridSequential(prefix='')
            self.layer4 = nn.HybridSequential(prefix='')
            self.layer5 = nn.HybridSequential(prefix='')

            with self.layer1.name_scope():
                self.layer1.add(nn.Conv2D(configs[1], kernel_size=11, strides=2))
                self.layer1.add(nn.BatchNorm())
                self.layer1.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.layer1.add(nn.Activation('relu'))

            with self.layer2.name_scope():
                self.layer2.add(nn.Conv2D(configs[2], kernel_size=5))
                self.layer2.add(nn.BatchNorm())
                self.layer2.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.layer2.add(nn.Activation('relu'))

            with self.layer3.name_scope():
                self.layer3.add(nn.Conv2D(configs[3], kernel_size=3))
                self.layer3.add(nn.BatchNorm())
                self.layer3.add(nn.Activation('relu'))

            with self.layer4.name_scope():

                self.layer4.add(nn.Conv2D(configs[4], kernel_size=3))
                self.layer4.add(nn.Activation('relu'))

            with self.layer5.name_scope():
                self.layer5.add(nn.Conv2D(configs[5], kernel_size=3))
                self.layer5.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

def alexnetlegacy(**kwargs):
    return AlexNetLegacy(**kwargs)

def alexnet(**kwargs):
    return AlexNet(**kwargs)