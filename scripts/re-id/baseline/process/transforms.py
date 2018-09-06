from __future__ import absolute_import
import os, numbers, random, math

from mxnet.gluon.data import dataset
from mxnet import image, nd
from mxnet.gluon import Block


class Pad(Block):
    def __init__(self, padding):
        super(Pad, self).__init__()
        self.padding = padding

    def forward(self, x):
        if not isinstance(self.padding, (numbers.Number, tuple)):
            raise TypeError('Got inappropriate padding arg')
        shape = x.shape
        if isinstance(self.padding, numbers.Number):
            res = nd.zeros((shape[0]+2*self.padding, shape[1]+2*self.padding, shape[2]))
            res[self.padding:shape[0]+self.padding,self.padding:shape[1]+self.padding, :] = x
        if isinstance(self.padding, tuple):
            res = nd.zeros((shape[0]+2*self.padding[0], shape[1]+2*self.padding[1], shape[2]))
            res[self.padding:shape[0]+self.padding[0], self.padding:shape[1]+self.padding[1], :] = x
        return res


class RandomCrop(Block):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, x):
        if not isinstance(self.size, (numbers.Number, tuple)):
            raise TypeError('Got inappropriate size arg')
        if isinstance(self.size, numbers.Number):
            size = (self.size, self.size)
        else:
            size = self.size
        return image.random_crop(x, size)[0]


class RandomErasing(Block):
    def __init__(self, probability = 0.5, s_min = 0.02, s_max = 0.4, ratio = 0.3, mean = (125.31, 122.96, 113.86)):
        super(RandomErasing, self).__init__()
        self.probability = probability
        self.mean = mean
        self.s_min = s_min
        self.s_max = s_max
        self.ratio = ratio

    def forward(self, x):
        if not isinstance(self.probability, numbers.Number):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.s_min, numbers.Number):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.s_max, numbers.Number):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.ratio, numbers.Number):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.mean, (numbers.Number, tuple)):
            raise TypeError('Got inappropriate size arg')

        if random.uniform(0, 1) > self.probability:
            return x

        width, height, _ = x.shape
        area = width * height
        target_area = random.uniform(self.s_min, self.s_max) * area
        aspect_ratio = random.uniform(self.ratio, 1/self.ratio)
        
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if w < width and h < height:
            x1 = random.randint(0, width - w)
            y1 = random.randint(0, height - h)
            x[x1:x1+w, y1:y1+h, 0] = self.mean[0]
            x[x1:x1+w, y1:y1+h, 1] = self.mean[1]
            x[x1:x1+w, y1:y1+h, 2] = self.mean[2]
        return x