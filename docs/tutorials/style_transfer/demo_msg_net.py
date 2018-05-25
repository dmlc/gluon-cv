"""1.Getting Started with Pre-trained MSG-Net
==========================================

.. image:: https://github.com/zhanghang1989/MSG-Net/blob/master/images/figure1.jpg
    :width: 55%
    :align: left



.. note::
    Hang Zhang, and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."::

        @article{zhang2017multistyle,
            title={Multi-style Generative Network for Real-time Transfer},
            author={Zhang, Hang and Dana, Kristin},
            journal={arXiv preprint arXiv:1703.06953},
            year={2017}
        }
"""
import mxnet as mx
import gluoncv
##############################################################################
# Get the content and style images
# --------------------------------
#
# download the example images
content_url = 'https://github.com/dmlc/web-data/blob/master/mxnet/example/' + \
    'style_transfer/images/content/venice-boat.jpg?raw=True'
style_url = 'https://github.com/dmlc/web-data/blob/master/mxnet/example/' + \
    'style_transfer/images/styles/candy.jpg?raw=True'
content_fn = 'venice-boat.jpg'
style_fn = 'candy.jpg'
gluoncv.utils.download(content_url, content_fn)
gluoncv.utils.download(style_url, style_fn)

# load the image and preprocessing using preset utils
from gluoncv.data.transforms.presets.msgnet import load_image, preprocess_batch
content_img = load_image(content_fn)
style_img = preprocess_batch(load_image(style_fn))

##############################################################################
# Get the pre-trained MSG-Net model and test style transfer
# ---------------------------------------------------------
#
# get pre-trained model
model = gluoncv.model_zoo.get_msgnet(pretrained=True)

model.set_target(style_img)
output = model(content_img)
gluoncv.data.transforms.presets.msgnet.save_bgrimage(output.squeeze(), 'transfered.jpg')

##############################################################################
# Display the style transfer result
# ---------------------------------
#
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# subplot 1 for img
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(mpimg.imread(content_fn))
# subplot 2 for the mask
fig.add_subplot(1,2,2)
plt.imshow(mpimg.imread('transfered.jpg'))
# display
plt.show()
