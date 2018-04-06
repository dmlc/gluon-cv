import os
import shutil
import numpy as np
from PIL import Image

import mxnet.ndarray as F

__all__ = ['save_checkpoint', 'get_mask', 'resize_image', 'pad_image', 'crop_image',
           'flip_image']


def save_checkpoint(net, args, is_best=False):
    directory = "runs/%s/%s/%s/" % (args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename='checkpoint.params'
    filename = directory + filename
    net.collect_params().save(filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.params')


def resize_image(img, h, w):
    return F.contrib.BilinearResize2D(img, height=h, width=w)


def pad_image(img, args, crop_size=480):
    b,c,h,w = img.shape
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(args.mean) / np.array(args.std)
    img_pad = mx.nd.zeros((b,c,h+padh,w+padw)).as_in_context(img.context)
    for i in range(c):
        img_pad[:,i,:,:] = F.squeeze(
            F.pad(img[:,i,:,:].expand_dims(1), 'constant', 
                  pad_width=(0,0,0,0,0,padh,0,padw),
                  constant_value = pad_values[i]
            ))
    assert(img_pad.shape[2]>=crop_size and img_pad.shape[3]>=crop_size)
    return img_pad


def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]


def flip_image(img):
    assert(img.ndim == 4)
    return img.flip(3)


def get_mask(npimg, dataset):
    # recovery boundary
    if dataset == 'pascal_voc' or dataset == 'pascal_aug':
        npimg[npimg==21] = 255
    # put colormap
    out_img = Image.fromarray(npimg.astype('uint8'))
    if dataset == 'ade20k':
        out_img.putpalette(adepallete)
    else:
        out_img.putpalette(vocpallete)
    return out_img


# ref https://github.com/dmlc/mxnet/blob/master/example/fcn-xs/image_segmentaion.py
def getvocpallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
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


vocpallete = getvocpallete(256)
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
