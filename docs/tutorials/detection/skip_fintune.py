"""10. Skip Finetuning by reusing part of pre-trained model
===========================================================

There is a dilemma that pre-trained public dataset detection models need finetuning
before we can apply them to our interested domain.
While it is still a chanllenging
task, in this tutorial we showcase a very interesting way to reuse pre-trained models.

Basically, you can grab a GluonCV pre-trained detection model and reset classes to a subset of
coco categories, and it will be instantly ready to use without any tuning.

First let's import some necessary libraries:
"""

from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an Faster RCNN model trained on COCO
# dataset with ResNet-50 backbone.

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)

######################################################################
# Pre-process an image
# --------------------
# Similar to faster rcnn inference tutorial, we grab and preprocess a demo image

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

######################################################################
# Reset classes to exactly what we want
# -------------------------------------
# Original COCO model has 80 classes
print('coco classes: ', net.classes)
net.reset_class(classes=['bicycle', 'backpack'], reuse_weights=['bicycle', 'backpack'])
# now net has 2 classes as desired
print('new classes: ', net.classes)

######################################################################
# Inference and display
# ---------------------

box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()

######################################################################
# More flexible mapping strategy for reusing old weights
# ------------------------------------------------------
# We also support dict for 1-to-1 class weights re-mapping
# So we can take advantage of this to remap some categories
net = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)
net.reset_class(classes=['spaceship'], reuse_weights={'spaceship':'bicycle'})
box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()

######################################################################
# The same story for different models
# --------------------------------------------------------
# We can apply this strategy to SSD, YOLO and Mask-RCNN models
# Now we can use mask rcnn and reset class to detect person only

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
net.reset_class(classes=['person'], reuse_weights=['person'])
ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
orig_img = utils.viz.plot_mask(orig_img, masks)

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=net.classes, ax=ax)
plt.show()

######################################################################
# Feel excited?
# --------------
# Stay tuned for more generalized detection models with much more category
# knowledges than COCO and Pascal VOC!
