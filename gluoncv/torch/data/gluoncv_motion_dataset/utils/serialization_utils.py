"""Utility functions"""
# pylint: disable=missing-docstring, arguments-differ, method-hidden
import json
import pickle
import pathlib
import numpy as np


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, pathlib.PurePath):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


def load_json(fname, *args, **kwargs):
    with open(fname, 'r') as fd:
        obj = json.load(fd, *args, **kwargs)
    return obj


def save_json(obj, fname, sort_keys=True, indent=4, separators=None,
              encoder=ComplexEncoder, *args, **kwargs):
    with open(fname, 'w') as fd:
        json.dump(obj, fd, indent=indent, sort_keys=sort_keys, cls=encoder,
                  separators=separators, *args, **kwargs)


def load_pickle(fname, **kwargs):
    with open(fname, 'rb') as fd:
        obj = pickle.load(fd, **kwargs)
    return obj


def save_pickle(obj, fname, protocol=None, **kwargs):
    with open(fname, 'wb') as fd:
        pickle.dump(obj, fd, protocol=protocol, **kwargs)


def round_floats_for_json(obj, ndigits=2, key_ndigits=None):
    """
    Tries to round all floats in obj in order to reduce json size.
    ndigits is the default number of digits to round to,
    key_ndigits allows you to override this for specific dictionary keys,
    though there is no concept of nested keys.
    It converts numpy arrays and iterables to lists,
    so it should only be used when serializing to json
    """
    if key_ndigits is None:
        key_ndigits = {}

    if isinstance(obj, np.floating):
        obj = float(obj)
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, float):
        obj = round(obj, ndigits)
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            this_ndigits = key_ndigits.get(k, ndigits)
            new_obj[k] = round_floats_for_json(v, this_ndigits, key_ndigits)
        return new_obj
    elif isinstance(obj, str):
        return obj
    else:
        try:
            return [round_floats_for_json(x, ndigits, key_ndigits) for x in obj]
        except TypeError:
            pass

    return obj


def get_encoder_with_float_rounding(ndigits=2, key_ndigits=None):
    class FloatingDigitEncoder(ComplexEncoder):
        def encode(self, obj):
            obj = round_floats_for_json(obj, ndigits, key_ndigits)
            return super().encode(obj)

        def default(self, obj):
            obj = super().default(obj)
            obj = round_floats_for_json(obj, ndigits, key_ndigits)
            return obj

    return FloatingDigitEncoder


Floating2DigitEncoder = get_encoder_with_float_rounding()
