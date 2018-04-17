"""Getting Started with FCN Pre-trained Models
===========================================

This is a quick demo of using GluonVision FCN model.
Please follow the `installation guide <../index.html>`_ to install MXNet and GluonVision if not yet.
This demo code can be downloaded from :download:`demo.py <../../scripts/segmentation/demo.py>`.
"""
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluonvision
# using cpu
ctx = mx.cpu(0)


##############################################################################
# Prepare the image 

# download the example image
url = 'https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/1.jpg'
filename = '1.jpg'
gluonvision.utils.download(url)

# load the image
with open(filename, 'rb') as f:
    img = image.imdecode(f.read())

##############################################################################
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/1.jpg
#    :width: 45%
#

# normalize the image using dataset mean
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
img = transform_fn(img)
img = img.expand_dims(0).as_in_context(ctx)

##############################################################################
# Load the pre-trained model and make prediction

# get pre-trained model
model = gluonvision.model_zoo.get_fcn_voc_resnet101(pretrained=True)
# make prediction using single scale
output = model.evaluate(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

##############################################################################
# Add color pallete for visualization
from utils import get_mask
mask = get_mask(predict, 'pascal_voc')
mask.save('output.png')


##############################################################################
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/1.png
#    :width: 45%
#

##############################################################################
# More Examples
#
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/4.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/4.png
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/5.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/5.png
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/6.jpg
#    :width: 45%
#
#.. image:: https://raw.githubusercontent.com/zhanghang1989/gluon-vision-figures/master/voc_examples/6.png
#    :width: 45%
