# pylint: disable=wildcard-import, unused-wildcard-import
"""
GluonCV-PyTorch model zoo
"""
from .action_recognition import *


__all__ = ['get_model', 'get_model_list']


_models = {
    'resnet18_v1b_kinetics400': resnet18_v1b_kinetics400,
    'resnet34_v1b_kinetics400': resnet34_v1b_kinetics400,
    'resnet50_v1b_kinetics400': resnet50_v1b_kinetics400,
    'resnet101_v1b_kinetics400': resnet101_v1b_kinetics400,
    'resnet152_v1b_kinetics400': resnet152_v1b_kinetics400,
    'resnet50_v1b_sthsthv2': resnet50_v1b_sthsthv2,
    'i3d_resnet50_v1_kinetics400': i3d_resnet50_v1_kinetics400,
    'i3d_resnet101_v1_kinetics400': i3d_resnet101_v1_kinetics400,
    'i3d_nl5_resnet50_v1_kinetics400': i3d_nl5_resnet50_v1_kinetics400,
    'i3d_nl10_resnet50_v1_kinetics400':i3d_nl10_resnet50_v1_kinetics400,
    'i3d_nl5_resnet101_v1_kinetics400': i3d_nl5_resnet101_v1_kinetics400,
    'i3d_nl10_resnet101_v1_kinetics400': i3d_nl10_resnet101_v1_kinetics400,
    'i3d_resnet50_v1_sthsthv2': i3d_resnet50_v1_sthsthv2,
    'slowfast_4x16_resnet50_kinetics400': slowfast_4x16_resnet50_kinetics400,
    'slowfast_8x8_resnet50_kinetics400': slowfast_8x8_resnet50_kinetics400,
    'slowfast_4x16_resnet101_kinetics400': slowfast_4x16_resnet101_kinetics400,
    'slowfast_8x8_resnet101_kinetics400': slowfast_8x8_resnet101_kinetics400,
    'slowfast_16x8_resnet101_kinetics400': slowfast_16x8_resnet101_kinetics400,
    'slowfast_16x8_resnet101_50_50_kinetics400': slowfast_16x8_resnet101_50_50_kinetics400,
    'slowfast_16x8_resnet50_sthsthv2': slowfast_16x8_resnet50_sthsthv2,
    'i3d_slow_resnet50_f32s2_kinetics400': i3d_slow_resnet50_f32s2_kinetics400,
    'i3d_slow_resnet50_f16s4_kinetics400': i3d_slow_resnet50_f16s4_kinetics400,
    'i3d_slow_resnet50_f8s8_kinetics400': i3d_slow_resnet50_f8s8_kinetics400,
    'i3d_slow_resnet101_f32s2_kinetics400': i3d_slow_resnet101_f32s2_kinetics400,
    'i3d_slow_resnet101_f16s4_kinetics400': i3d_slow_resnet101_f16s4_kinetics400,
    'i3d_slow_resnet101_f8s8_kinetics400': i3d_slow_resnet101_f8s8_kinetics400,
    'r2plus1d_v1_resnet18_kinetics400': r2plus1d_v1_resnet18_kinetics400,
    'r2plus1d_v1_resnet34_kinetics400': r2plus1d_v1_resnet34_kinetics400,
    'r2plus1d_v1_resnet50_kinetics400': r2plus1d_v1_resnet50_kinetics400,
    'r2plus1d_v1_resnet101_kinetics400': r2plus1d_v1_resnet101_kinetics400,
    'r2plus1d_v1_resnet152_kinetics400': r2plus1d_v1_resnet152_kinetics400,
    'r2plus1d_v2_resnet152_kinetics400': r2plus1d_v2_resnet152_kinetics400,
    'ircsn_v2_resnet152_f32s2_kinetics400': ircsn_v2_resnet152_f32s2_kinetics400,
    'tpn_resnet50_f8s8_kinetics400': tpn_resnet50_f8s8_kinetics400,
    'tpn_resnet50_f16s4_kinetics400': tpn_resnet50_f16s4_kinetics400,
    'tpn_resnet50_f32s2_kinetics400': tpn_resnet50_f32s2_kinetics400,
    'tpn_resnet101_f8s8_kinetics400': tpn_resnet101_f8s8_kinetics400,
    'tpn_resnet101_f16s4_kinetics400': tpn_resnet101_f16s4_kinetics400,
    'tpn_resnet101_f32s2_kinetics400': tpn_resnet101_f32s2_kinetics400,
}


def get_model(cfg):
    """Returns a pre-defined model by name

    Returns
    -------
    The model.
    """
    name = cfg.CONFIG.MODEL.NAME.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](cfg)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()
