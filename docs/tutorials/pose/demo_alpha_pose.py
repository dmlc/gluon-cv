"""2. Predict with pre-trained AlphaPose Estimation models
==========================================================

This article shows how to play with pre-trained Alpha Pose models with only a few
lines of code.

First let's import some necessary libraries:
"""

from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get a Alpha Pose model trained with input images of size 256x192 on MS COCO
# dataset. We pick the one using ResNet-101 V1b as the base model. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
#
# Note that a Alpha Pose model takes a top-down strategy to estimate
# human pose in detected bounding boxes from an object detection model.

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

detector.reset_class(["person"], reuse_weights=['person'])

######################################################################
# Pre-process an image for detector, and make inference
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. Here we
# specify that we resize the short edge of the image to 512 px. But you can
# feed an arbitrarily sized image.
#
# This function returns two results. The first is a NDArray with shape
# ``(batch_size, RGB_channels, height, width)``. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/pose/soccer.png?raw=true',
                          path='soccer.png')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

class_IDs, scores, bounding_boxs = detector(x)

######################################################################
# Process tensor from detector to keypoint network
# --------------------
#
# Next we process the output from the detector.
#
# For a Alpha Pose network, it expects the input has the size 256x192,
# and the human is centered. We crop the bounding boxed area
# for each human, and resize it to 256x192, then finally normalize it.
#
# In order to make sure the bounding box has included the entire person,
# we usually slightly upscale the box size.

pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs)

######################################################################
# Predict with a Alpha Pose network
# --------------------
#
# Now we can make prediction.
#
# A Alpha Pose network predicts the heatmap for each joint (i.e. keypoint).
# After the inference we search for the highest value in the heatmap and map it to the
# coordinates on the original image.

predicted_heatmap = pose_net(pose_input)
pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

######################################################################
# Display the pose estimation results
# ---------------------
#
# We can use :py:func:`gluoncv.utils.viz.plot_keypoints` to visualize the
# results.

ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
plt.show()
