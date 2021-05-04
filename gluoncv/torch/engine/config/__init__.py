"""The global configs registry"""
from .action_recognition import _C as _C_action_recognition
from .coot import _C as _C_coot
from .directpose import _C as _C_directpose

__all__ = ['get_cfg_defaults']

_CONFIG_REG = {
    "action_recognition": _C_action_recognition,
    "coot": _C_coot,
    "directpose": _C_directpose
}

def get_cfg_defaults(name='action_recognition'):
    """Get a yacs CfgNode object with default values for by name.

    Parameters
    ----------
    name : str
        The name of the root config, e.g. action_recognition, coot, directpose...

    Returns
    -------
    yacs.CfgNode object

    """
    assert isinstance(name, str), f"{name} must be a str"
    name = name.lower()
    if name not in _CONFIG_REG.keys():
        raise ValueError(f"Unknown root config with name: {name}")
    return _CONFIG_REG[name].clone()
