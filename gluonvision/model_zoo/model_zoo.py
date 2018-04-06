"""Model store which handles pretrained models from both
mxnet.gluon.model_zoo.vision and gluonvision.models
"""
from mxnet import gluon
from .ssd import *

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    models = {
        'ssd_300_vgg16_atrous': ssd_300_vgg16_atrous,
        'ssd_512_vgg16_atrous': ssd_512_vgg16_atrous,
        'ssd_512_resnet18_v1': ssd_512_resnet18_v1,
        'ssd_512_resnet50_v1': ssd_512_resnet50_v1,
        }
    try:
        net = gluon.model_zoo.vision.get_model(name, **kwargs)
    except ValueError as e:
        name = name.lower()
        if name not in models:
            raise ValueError('%s\n\t%s' % (str(e), '\n\t'.join(sorted(models.keys()))))
        net = models[name](**kwargs)
    return net
