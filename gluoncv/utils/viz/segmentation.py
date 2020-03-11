"""Segmentation Utils"""
from PIL import Image
import mxnet as mx
from mxnet.gluon import HybridBlock

__all__ = ['get_color_pallete', 'DeNormalize']

def get_color_pallete(npimg, dataset='pascal_voc'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')

    Returns
    -------
    out_img : PIL.Image
        Image with color pallete

    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255
    # put colormap
    if dataset == 'ade20k':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(adepallete)
        return out_img
    elif dataset == 'citys':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityspallete)
        return out_img
    elif dataset == 'mhpv1':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(mhpv1pallete)
        return out_img
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(vocpallete)
    return out_img


class DeNormalize(HybridBlock):
    """Denormalize the image"""
    # pylint: disable=arguments-differ,unused-argument
    def __init__(self, mean, std):
        super(DeNormalize, self).__init__()
        self.mean = mx.nd.array(mean, ctx=mx.cpu(0))
        self.std = mx.nd.array(std, ctx=mx.cpu(0))

    def hybrid_forward(self, F, x):
        return x * self.std .reshape(shape=(3, 1, 1)) + self.mean.reshape(shape=(3, 1, 1))


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0, n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete

vocpallete = _getvocpallete(256)

# pylint: disable=bad-whitespace

"""
The following numerical list is the color palette when visualizing a semantic segmentation mask.
Every three numbers is a RGB combination, corresponding to a specific color for each class.
For example, [0,0,0] is black indicating background. [120,120,120] is gray indicating wall.

For complete information, please see the color encoding table for ADE20K at:
https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit#gid=0

Please see the color encoding table for Cityscapes at:
https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
"""

adepallete = [
    0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,
    5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,
    143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,
    7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,
    10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,
    20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,
    20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,
    31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,
    0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,
    0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,
    31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,
    194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,
    0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,
    0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,
    163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,
    10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,
    255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,
    133,255,255,214,0,25,194,194,102,255,0,92,0,255]

cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

mhpv1pallete = [
    255, 255, 255,
    165, 42, 42,
    255, 0, 0,
    0, 128, 0,
    165, 42, 42,
    255, 69, 0,
    255, 20, 147,
    30, 144, 255,
    85, 107, 47,
    0, 128, 128,
    139, 69, 19,
    70, 130, 180,
    50, 205, 50,
    0, 0, 205,
    0, 191, 255,
    0, 255, 255,
    0, 250, 154,
    173, 255, 47,
    255, 255, 0,
]
