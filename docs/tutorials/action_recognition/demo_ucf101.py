"""1. Getting Started with Pre-trained TSN Model on UCF101
======================================================

`UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_  is an action recognition dataset
of realistic action videos, collected from YouTube. With 13,320 short trimmed videos
from 101 action categories, it is one of the most widely used dataset in the research
community for benchmarking state-of-the-art video action recognition models.

In this tutorial, we will demonstrate how to load a pre-trained model from :ref:`gluoncv-model-zoo`
and classify video frames from the Internet or your local disk.

Step by Step
------------------

Let's first try out a pre-trained UCF101 model with a few lines of python code.

First, please follow the `installation guide <../../index.html#installation>`__
to install ``MXNet`` and ``GluonCV`` if you haven't done so yet.
"""

import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model

################################################################
#
# Then, we download and show the example image:

url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/ThrowDiscus.png'
im_fname = utils.download(url)

img = image.imread(im_fname)

plt.imshow(img.asnumpy())
plt.show()

################################################################
# In case you don't recognize it, the image is a man throwing discus. :)
#
# Now we define transformations for the image.

transform_fn = transforms.Compose([
    video.VideoCenterCrop(size=224),
    video.VideoToTensor(),
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

################################################################
# This transformation function does three things:
# center crop the image to 224x224 in size,
# transpose it to ``num_channels*height*width``,
# and normalize with mean and standard deviation calculated across all ImageNet images.
#
# What does the transformed image look like?

img = transform_fn(img)
plt.imshow(nd.transpose(img, (1,2,0)).asnumpy())
plt.show()

################################################################
# Can't recognize anything? *Don't panic!* Neither do I.
# The transformation makes it more "model-friendly", instead of "human-friendly".
#
# Next, we load a pre-trained model.

net = get_model('vgg16_ucf101', nclass=101, pretrained=True)

################################################################
#
# Finally, we prepare the image and feed it to the model

pred = net(img.expand_dims(axis=0))

classes = net.classes
topK = 5
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input video frame is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.'%
          (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))

################################################################
#
# We can see that our pre-trained model predicts this video frame
# to be ``throw discus`` action with high confidence.

################################################################
# Next Step
# ---------
#
# If you would like to dive deeper into training TSN models on ``UCF101``,
# feel free to read the next `tutorial on UCF101 <dive_deep_ucf101.html>`__.
