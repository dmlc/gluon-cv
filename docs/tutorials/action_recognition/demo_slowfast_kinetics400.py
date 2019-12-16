"""3. Getting Started with Pre-trained SlowFast Models on Kinetcis400
=====================================================================

`Kinetics400 <https://deepmind.com/research/open-source/kinetics>`_  is an action recognition dataset
of realistic action videos, collected from YouTube. With 306,245 short trimmed videos
from 400 action categories, it is one of the largest and most widely used dataset in the research
community for benchmarking state-of-the-art video action recognition models.

In this tutorial, we will demonstrate how to load a pre-trained SlowFast model from :ref:`gluoncv-model-zoo`
and classify a video clip from the Internet or your local disk into one of the 400 action classes.

Step by Step
------------

We will try out a pre-trained SlowFast model on a single video clip.

First, please follow the `installation guide <../../index.html#installation>`__
to install ``MXNet`` and ``GluonCV`` if you haven't done so yet.
"""

import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model

################################################################
# Then, we download the video and extract a 32-frame clip from it.

from gluoncv.utils.filesystem import try_import_decord
decord = try_import_decord()

url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
video_fname = utils.download(url)
vr = decord.VideoReader(video_fname)
fast_frame_id_list = range(0, 64, 2)
slow_frame_id_list = range(0, 64, 16)
frame_id_list = list(fast_frame_id_list) + list(slow_frame_id_list)
video_data = vr.get_batch(frame_id_list).asnumpy()
clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

################################################################
# Now we define transformations for the video clip.
# This transformation function does three things:
# center crop the image to 224x224 in size,
# transpose it to ``num_channels*num_frames*height*width``,
# and normalize with mean and standard deviation calculated across all ImageNet images.

transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
clip_input = transform_fn(clip_input)
clip_input = np.stack(clip_input, axis=0)
clip_input = clip_input.reshape((-1,) + (36, 3, 224, 224))
clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
print('Video data is downloaded and preprocessed.')

################################################################
# Next, we load a pre-trained I3D model.

model_name = 'slowfast_4x16_resnet50_kinetics400'
net = get_model(model_name, nclass=400, pretrained=True)
print('%s model is successfully loaded.' % model_name)

################################################################
# Finally, we prepare the video clip and feed it to the model.

pred = net(nd.array(clip_input))

classes = net.classes
topK = 5
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input video clip is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.'%
          (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))

################################################################
#
# We can see that our pre-trained model predicts this video clip
# to be ``abseiling`` action with high confidence.

################################################################
# Next Step
# ---------
#
# If you would like to dive deeper into training SlowFast models on ``Kinetics400``,
# feel free to read the next `tutorial on Kinetics400 <dive_deep_slowfast_kinetics400.html>`__.
