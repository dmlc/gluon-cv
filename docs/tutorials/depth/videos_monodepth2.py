"""02. Predict depth from a image sequence or a video with pre-trained Monodepth2 models
===========================================================================
This article will demonstrate how to estimate depth from your image sequence or video stream.

Please follow the `installation guide <../../index.html#installation>`__
to install MXNet and GluonCV if not yet.

First, import the necessary modules.
"""
import os
import argparse
import time
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth

import matplotlib as mpl
import matplotlib.cm as cm
import cv2

# using cpu
ctx = mx.cpu(0)

##############################################################################
# Prepare the data
# -----------------
#
# In this tutorial, we use one sequence of KITTI RAW dataset as example. Because KITTI RAW dataset only provide image
# sequences, the input format is image sequences in this tutorials.
#
# Follow the command to download example data::
#
#       cd ~/.mxnet/datasets/kitti/examples
#       wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip
#       unzip 2011_09_26_drive_0095_sync.zip
#
#
# After getting the dataset, we can easily loading images with PIL.
data_path = "~/.mxnet/datasets/kitti/example/2011_09_26/2011_09_26_drive_0095_sync/image_02/data"

files = os.listdir(data_path)
files.sort()

raw_img_sequences = []
for file in files:
    file = os.path.join(data_path, file)
    img = pil.open(file).convert('RGB')
    raw_img_sequences.append(img)

original_width, original_height = raw_img_sequences[0].size

##############################################################################
# Loading the model
# -----------------
#
# In this tutorial we feed frames from the image sequences into a depth estimation model,
# then we could get the depth map of the input frame.
#
# For the model we use monodepth2_resnet18_kitti_mono_stereo_640x192 as it is accurate and
# could be recovery scaling factor of stereo baseline.

model_zoo = 'monodepth2_resnet18_kitti_mono_stereo_640x192'
model = gluoncv.model_zoo.get_model(model_zoo, pretrained_base=False, ctx=ctx, pretrained=True)

##############################################################################
# Estimation loop
# -----------------
#
# For each frame, we perform the following steps:
#
# - loading a frame from the image sequence
# - pre-process the image
# - estimate the disparity for the image
# - transfer the disparity to a depth map
# - store the depth map to the prediction sequence

min_depth = 0.1
max_depth = 100

# while use stereo or mono+stereo model, we could get real depth value
scale_factor = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 80

feed_height = 192
feed_width = 640

pred_depth_sequences = []
pred_disp_sequences = []
for img in raw_img_sequences:
    img = img.resize((feed_width, feed_height), pil.LANCZOS)
    img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

    outputs = model.predict(img)
    mx.nd.waitall()
    pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)
    t = time.time()
    pred_disp = pred_disp.squeeze().as_in_context(mx.cpu()).asnumpy()
    pred_disp = cv2.resize(src=pred_disp, dsize=(original_width, original_height))
    pred_disp_sequences.append(pred_disp)

    pred_depth = 1 / pred_disp
    pred_depth *= scale_factor
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    pred_depth_sequences.append(pred_depth)


##############################################################################
# Visualization and store results
# -----------------
#
# Here, we provide the example of store the prediction results. Including:
# - store depth map
output_path = os.path.join(os.path.expanduser("."), "tmp")

pred_path = os.path.join(output_path, 'pred')
if not os.path.exists(pred_path):
    os.makedirs(pred_path)

for pred, file in zip(pred_depth_sequences, files):
    pred_out_file = os.path.join(pred_path, file)
    cv2.imwrite(pred_out_file, pred)

##############################################################################
# - store disparity and save it to a video
rgb_path = os.path.join(output_path, 'rgb')
if not os.path.exists(rgb_path):
    os.makedirs(rgb_path)

output_sequences = []
for raw_img, pred, file in zip(raw_img_sequences, pred_disp_sequences, files):
    vmax = np.percentile(pred, 95)
    normalizer = mpl.colors.Normalize(vmin=pred.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(pred)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    raw_img = np.array(raw_img)
    pred = np.array(im)
    output = np.concatenate((raw_img, pred), axis=0)
    output_sequences.append(output)

    pred_out_file = os.path.join(rgb_path, file)
    cv2.imwrite(pred_out_file, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

width = int(output_sequences[0].shape[1] + 0.5)
height = int(output_sequences[0].shape[0] + 0.5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    os.path.join(output_path, 'demo.mp4'), fourcc, 20.0, (width, height))

for frame in output_sequences:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    out.write(frame)
    cv2.imshow('demo', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

##############################################################################
# We release the webcam before exiting:
out.release()
cv2.destroyAllWindows()


##############################################################################
# You can start with example code.
# -----------------
#
# Download the script to run the demo
#
# :download:`Download cam_demo.py<../../../scripts/depth/demo.py>`
#
# Example running command::
#
#         python demo.py --model_zoo monodepth2_resnet18_kitti_mono_stereo_640x192 --input_format image --data_path ~/.mxnet/datasets/kitti/example/2011_09_26/2011_09_26_drive_0095_sync/image_02/data --output_format video
#
# For more demo command options, please run ``python demo.py -h``

#  .. hint::
#
#     This tutorial directly loads the image sequence or video into a list,
#     so it cannot work when the image sequence or video is large.
#     Here is just provide a example about using pretrained monodepth2 model to do prediction for users.
#


