"""Common training hyperparameters"""
from sacred import Ingredient

train = Ingredient('train')


@train.config
def cfg():
    gpus = (0, 1, 2, 3)  # gpu individual ids, not necessarily consecutive
    num_workers = 16  # cpu workers, the larger the more processes used
    batch_size = 32
    epochs = 3
    resume = ''
    auto_resume = True  # try to automatically resume last trial if config is default
    start_epoch = 0
    momentum = 0.9  # SGD momentum
    wd = 1e-4  # weight decay
    save_interval = 10  # Saving parameters epoch interval, best model will always be saved
    log_interval = 100  # logging interval


validation = Ingredient('validation')


@validation.config
def cfg():
    num_workers = 32  # cpu workers, the larger the more processes used
    batch_size = 32  # validation batch size
    interval = 10  # validation epoch interval, for slow validations
