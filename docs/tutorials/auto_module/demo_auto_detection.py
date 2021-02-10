"""04. HPO with Faster R-CNN end-to-end on PASCAL VOC
=====================================================

This tutorial goes through the basic steps of using AutoGluon to tune hyper-parameters for a
Faster-RCNN [Ren15]_ object detection model provided by GluonCV.
"""

##########################################################
# Dataset
# -------
#
# Please first go through this :ref:`sphx_glr_build_examples_datasets_pascal_voc.py` tutorial to setup Pascal
# VOC dataset on your disk.
# Then, we are ready to load training and validation images.

import autogluon.core as ag

from gluoncv.auto.estimators.faster_rcnn import FasterRCNNEstimator
from gluoncv.auto.tasks.object_detection import ObjectDetection

# Define search space
time_limits = 60 * 60  # 1hr
search_args = {'dataset': 'voc', 'split_ratio': 0.8, 'num_trials': 30,
               'epochs': ag.Categorical(30, 40, 50, 60), 'num_workers': 16,
               'net': ag.Categorical('resnest101', 'resnest50'), 'meta_arch': 'faster_rcnn',
               'search_strategy': 'random', 'search_options': {},
               'lr': ag.Categorical(0.005, 0.002, 2e-4, 5e-4), 'transfer': False,
               'data_shape': (640, 800), 'nthreads_per_trial': 12, 'verbose': False,
               'ngpus_per_trial': 4, 'batch_size': 4, 'hybridize': True,
               'lr_decay_epoch': ag.Categorical([24, 28], [35], [50, 55], [40], [45], [55],
                                                [30, 35], [20]),
               'warmup_iters': ag.Int(5, 500), 'resume': False, 'checkpoint': 'checkpoint/exp1.ag',
               'visualizer': 'none', 'start_epoch': 0, 'lr_mode': 'step', 'lr_decay': 0.1,
               'lr_decay_period': 0, 'warmup_lr': 0.0, 'warmup_epochs': 2, 'warmup_factor': 1. / 3.,
               'momentum': 0.9, 'log_interval': 100, 'save_prefix': '', 'save_interval': 10,
               'val_interval': 1, 'num_samples': -1, 'no_random_shape': False, 'no_wd': False,
               'mixup': False, 'no_mixup_epochs': 20, 'reuse_pred_weights': True, 'horovod': False,
               'grace_period': None, 'auto_search': True, 'seed': 223,
               'wd': ag.Categorical(1e-4, 5e-4, 2.5e-4), 'syncbn': ag.Bool(), 'label_smooth': False,
               'time_limits': time_limits, 'dist_ip_addrs': []}

# Construct a object detection task based on the config.
# task = ObjectDetection(search_args, FasterRCNNEstimator)
#
# # Automatically fit a model.
# estimator = task.fit()
#
# # Evaluate the final model on test set.
# test_map = estimator.evaluate()
# print("mAP on test dataset: {}".format(test_map[-1][-1]))
# print(test_map)
#
# # Save our final model.
# estimator.save('final_model.model')

##########################################################
# References
# ----------
#
# .. [Girshick14] Ross Girshick and Jeff Donahue and Trevor Darrell and Jitendra Malik. Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. CVPR 2014.
# .. [Girshick15] Ross Girshick. Fast {R-CNN}. ICCV 2015.
# .. [Ren15] Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun. Faster {R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.
# .. [He16] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
# .. [Lin17] Tsung-Yi Lin and Piotr Dollár and Ross Girshick and Kaiming He and Bharath Hariharan and Serge Belongie. Feature Pyramid Networks for Object Detection. CVPR 2017.
