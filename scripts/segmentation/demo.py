"""Getting Started with FCN Pre-trained Models
===========================================

Tutorial and Examples
_____________________
test
"""
import os
import argparse
from PIL import Image
from tqdm import tqdm
import mxnet as mx
import mxnet.ndarray as F

from gluonvision.model_zoo import FCN

from utils.utils import get_mask
from utils.data_utils import Compose, ToTensor, Normalize

def main():
    # this is a toy example of using gluonvision for semantic segmentation
    ctx = mx.cpu(0)

    # read image and normalize the data
    transform = Compose([
        ToTensor(ctx=ctx),
        Normalize(mean=[.485, .456, .406], std=[.229, .224, .225], ctx=ctx)])
    image = load_image('examples/1.jpg', transform, ctx)

    # load model
    model = FCN(nclass=22, backbone='resnet50')
    model.load_params('fcn50_voc.params', ctx=ctx)

    test_image(image, model)
    #test_image_folder('valimg', 'outmasks', transform, model, ctx)

##############################################################################
#
#.. image:: examples/1.jpg
#    :width: 45%
#
#.. image:: examples/1.png
#    :width: 45%
#
#.. image:: examples/4.jpg
#    :width: 45%
#
#.. image:: examples/4.png
#    :width: 45%
#
#.. image:: examples/5.jpg
#    :width: 45%
#
#.. image:: examples/5.png
#    :width: 45%
#
#.. image:: examples/6.jpg
#    :width: 45%
#
#.. image:: examples/6.png
#    :width: 45%




def test_image(image, model):
    # make prediction using single scale
    output = model(image)
    predict = F.squeeze(F.argmax(output, 1)).asnumpy()

    # add color pallete for visualization
    mask = get_mask(predict, 'pascal_voc')
    mask.save('output.png')


def test_image_folder(img_folder, out_folder, transform, model, ctx):
    img_paths = get_folder_images(img_folder)
    for path in tqdm(img_paths):
        image = load_image(path, transform, ctx)
        # make prediction using single scale
        output = model(image)
        predict = F.squeeze(F.argmax(output, 1)).asnumpy()
        # add color pallete for visualization
        mask = get_mask(predict, 'pascal_voc')
        outname = os.path.splitext(os.path.basename(path))[0] + '.png'
        mask.save(os.path.join(out_folder, outname))


def load_image(path, transform, ctx):
    image = Image.open(path).convert('RGB')
    image = transform(image)
    image = image.expand_dims(0).as_in_context(ctx)
    return image


def get_folder_images(img_folder):
    img_paths = []  
    for filename in os.listdir(img_folder):
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            img_paths.append(imgpath)
    return img_paths


if __name__ == "__main__":
    main()
