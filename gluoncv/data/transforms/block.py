# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
# pylint: disable= missing-docstring
"Addtional image transforms."

import random
import math
import numpy as np
from mxnet import image, nd
from mxnet.gluon import Block


__all__ = ['RandomCrop', 'RandomErasing']


class RandomCrop(Block):
    """Randomly crop `src` with `size` (width, height).
    Padding is optional.
    Upsample result if `src` is smaller than `size`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of the final output.
    pad: int or tuple
        if int, size of the zero-padding
        if tuple, number of values padded to the edges of each axis.
            ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
            ((before, after),) yields same before and after pad for each axis.
            (pad,) or int is a shortcut for before = after = pad width for all axes.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
    Outputs:
        - **out**: output tensor with (size[0] x size[1] x C) or (size x size x C) shape.
    """

    def __init__(self, size, pad=None, interpolation=2):
        super(RandomCrop, self).__init__()
        numeric_types = (float, int, np.generic)
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, interpolation)
        self.pad = ((pad, pad), (pad, pad), (0, 0)) if isinstance(pad, int) else pad
    def forward(self, x):
        if self.pad:
            return image.random_crop(nd.array(
                np.pad(x.asnumpy(), self.pad, mode='constant', constant_values=0)), *self._args)[0]
        else:
            return image.random_crop(x, *self._args)[0]

class RandomErasing(Block):
    """Randomly erasing the area in `src` between `s_min` and `s_max` with `probability`.
    `ratio` controls the ratio between width and height.
    `mean` means the value in erasing area.

    Parameters
    ----------
    probability : float
        Probability of erasing.
    s_min : float
        Min area to all area.
    s_max : float
        Max area to all area.
    ratio : float
        The ratio between width and height.
    mean : int or tuple of (R, G, B)
        The value in erasing area.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
    Outputs:
        - **out**: output tensor with (Hi x Wi x C) shape.
    """

    def __init__(self, probability=0.5, s_min=0.02, s_max=0.4, ratio=0.3,
                 mean=(125.31, 122.96, 113.86)):
        super(RandomErasing, self).__init__()
        self.probability = probability
        self.mean = mean
        self.s_min = s_min
        self.s_max = s_max
        self.ratio = ratio

    def forward(self, x):
        if not isinstance(self.probability, float):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.s_min, float):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.s_max, float):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.ratio, float):
            raise TypeError('Got inappropriate size arg')
        if not isinstance(self.mean, (int, tuple)):
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
