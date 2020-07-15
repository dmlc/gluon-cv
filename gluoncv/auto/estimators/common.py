"""Common training hyperparameters"""
from sacred import Ingredient

logging = Ingredient('logging')


@logging.config
def cfg():
    # Output directory for all training/validation artifacts.
    logdir = None
