"""Common training hyperparameters"""
from sacred import Ingredient

train_hyperparams = Ingredient('train_hyperparams')

@train_hyperparams.config
def cfg():
    gpus = (0, 1, 2, 3)          # gpu individual ids, not necessarily consecutive
    num_workers = 16             # cpu workers, the larger the more processes used
    pretrained_base = True       # use pre-trained weights from ImageNet
    transfer_from = None
    batch_size = 32
    epochs = 3
    resume = ''
    start_epoch = 0
