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
"Addtional image transforms."

from mxnet import image
from mxnet.gluon import Block
import numpy as np

__all__ = ['RandomResizedPadCrop']

class RandomResizedPadCrop(Block):
    """Crop the input image with random scale and aspect ratio.
    Makes a crop of the original image with random size (default: 0.08
    to 1.0 of the original image size) and random aspect ratio (default:
    3/4 to 4/3), then resize it to the specified size.
    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of the final output.
    pad: int
        Size of the zero-padding
    scale : tuple of two floats
        If scale is `(min_area, max_area)`, the cropped image's area will
        range from min_area to max_area of the original image's area
    ratio : tuple of two floats
        Range of aspect ratio of the cropped image before resizing.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
    Outputs:
        - **out**: output tensor with ((H+2*pad) x (W+2*pad) x C) shape.
    """

    def __init__(self, size, pad=0, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0),
                 interpolation=2):
        super(RandomResizedPadCrop, self).__init__()
        numeric_types = (float, int, np.generic)
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size, scale[0], ratio, interpolation)
        self.pad = pad

    def forward(self, x):
        if self.pad > 0:
            import pdb; pdb.set_trace()
            pad_tuple = (0, 0, 0, 0, self.pad, self.pad, self.pad, self.pad, 0, 0)
            x = x.expand_dims(axis=0).expand_dims(axis=0)
            x = x.pad(mode='constant', constant_value=0, pad_width=pad_tuple)
            x = x.squeeze(axis=(0, 1))

        return image.random_size_crop(x, *self._args)[0]
