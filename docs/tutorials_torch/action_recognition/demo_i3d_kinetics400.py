"""1. Getting Started with Pre-trained I3D Models on Kinetcis400
================================================================

`Kinetics400 <https://deepmind.com/research/open-source/kinetics>`_  is an action recognition dataset
of realistic action videos, collected from YouTube. With 306,245 short trimmed videos
from 400 action categories, it is one of the largest and most widely used dataset in the research
community for benchmarking state-of-the-art video action recognition models.

`I3D <https://arxiv.org/abs/1705.07750>`_ (Inflated 3D Networks) is a widely adopted 3D video
classification network. It uses 3D convolution to learn spatiotemporal information directly from videos.
I3D is proposed to improve `C3D <https://arxiv.org/abs/1412.0767>`_ (Convolutional 3D Networks) by inflating from 2D models.
We can not only reuse the 2D models' architecture (e.g., ResNet, Inception), but also bootstrap
the model weights from 2D pretrained models. In this manner, training 3D networks for video
classification is feasible and getting much better results.

In this tutorial, we will demonstrate how to load a pre-trained I3D model from :ref:`gluoncv-model-zoo`
and classify a video clip from the Internet or your local disk into one of the 400 action classes.

Step by Step
------------

We will try out a pre-trained I3D model on a single video clip.

First, please follow the `installation guide <../../index.html#installation>`__
to install ``PyTorch`` and ``GluonCV`` if you haven't done so yet.
"""

import numpy as np
import decord
import torch

from gluoncv.torch.utils.model_utils import download
from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model


################################################################
# Then, we download a video and extract a 32-frame clip from it.


url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
video_fname = download(url)
vr = decord.VideoReader(video_fname)
frame_id_list = range(0, 64, 2)
video_data = vr.get_batch(frame_id_list).asnumpy()


################################################################
# Now we define transformations for the video clip.
# This transformation function does four things:
# (1) resize the shorter side of video clip to short_side_size,
# (2) center crop the video clip to crop_size x crop_size,
# (3) transpose the video clip to ``num_channels*num_frames*height*width``,
# and (4) normalize it with mean and standard deviation calculated across all ImageNet images.


crop_size = 224
short_side_size = 256
transform_fn = video_transforms.Compose([video_transforms.Resize(short_side_size, interpolation='bilinear'),
                                         video_transforms.CenterCrop(size=(crop_size, crop_size)),
                                         volume_transforms.ClipToTensor(),
                                         video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


clip_input = transform_fn(video_data)
print('Video data is downloaded and preprocessed.')


################################################################
# Next, we load a pre-trained I3D model. Make sure to change the ``pretrained`` in the configuration file to True.


config_file = '../../../scripts/action-recognition/configuration/i3d_resnet50_v1_kinetics400.yaml'
cfg = get_cfg_defaults()
cfg.merge_from_file(config_file)
model = get_model(cfg)
model.eval()
print('%s model is successfully loaded.' % cfg.CONFIG.MODEL.NAME)


################################################################
# Finally, we prepare the video clip and feed it to the model.


with torch.no_grad():
    pred = model(torch.unsqueeze(clip_input, dim=0)).numpy()
print('The input video clip is classified to be class %d' % (np.argmax(pred)))


################################################################
# We can see that our pre-trained model predicts this video clip
# to be ``abseiling`` action with high confidence.

################################################################
# Next Step
# ---------
#
# If you would like to dive deeper into finetuing SOTA video models on your datasets,
# feel free to read the next `tutorial on finetuning <finetune_custom.html>`__.
