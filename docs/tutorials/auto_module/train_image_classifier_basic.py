"""02. Train Image Classification with Auto Estimator
=====================================================

This tutorial goes through the basic steps of using GluonCV auto estimator to train an image classifier
with custom hyper-parameters.
"""

##########################################################
# Train with default configurations
# ---------------------------------
from gluoncv.auto.estimators import ImageClassificationEstimator
from gluoncv.auto.tasks import ImageClassification

##########################################################
# In this tutorial, we use a small sample dataset
train, _, test = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
train, val, _ = train.random_split(val_size=0.1, test_size=0)

##########################################################
# Create an estimator with default configurations.
# We only change the number of GPUs to reflect the hardware constraint.
#
# Note that you may still launch training if no gpu is available(with {'gpus': []})
# but be prepared that this is painfully slow and not even possible to finish
# in case the dataset isn't tiny.
#
# We recommend that you use at least one nvidia gpu with more than 6GB free GPU memory.
classifier = ImageClassification({'gpus': [0]})


##########################################################
# run fit on train/validation data
classifier.fit(train, val)

##########################################################
# Evaluate the final model on test set
eval_maps = classifier.evaluate(test)
print("mAP on test dataset: {}".format(eval_maps[-1][-1]))

##########################################################
# save to/from disk to be used later
classifier.save('classifier.pkl')
classifier2 = ImageClassification.load('classifier.pkl')

##########################################################
# run prediction on test images
pred = classifier2.predict(test.iloc[0]['image'])
print('GroundTruth:', test.iloc[0])
print('prediction:', pred)

##########################################################
# Customize your training configurations
# --------------------------------------
# You may modify configurations to customize your training,
# supported fields:
print(ImageClassification._default_cfg)

##########################################################
# For example, we could change the learning rate and batch size
new_classifier = ImageClassification({'gpus': [0], 'train': {'batch_size': 16, 'lr': 0.01}})

##########################################################
# A more human readable format for modifying individual hyperparameter
# is to edit the yaml file saved automatically in `log_dir`
