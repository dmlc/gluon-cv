# pylint: disable=line-too-long
"""Encode/Decode helper function for compressed quantized models"""
import zlib
import base64

def encode_json(json_file, is_print=False):
    r""" Encode json string to compressed base64 string.

    Parameters
    ----------
    json_file : str
        String value represents the path to json file.
    is_print : bool
        Boolean value controls whether to print the encoded base64 string.
    """
    with open(json_file, encoding='utf-8') as fh:
        data = fh.read()
    zipped_str = zlib.compress(data.encode('utf-8'))
    b64_str = base64.b64encode(zipped_str)
    if is_print:
        print(b64_str)
    return b64_str

def decode_b64(b64_str, is_print=False):
    r""" Decode b64 string to json format

    Parameters
    ---------
    b64_str: str
        String value represents the compressed base64 string.
    is_print : bool
        Boolean value controls whether to print the decoded json string.
    """
    json_str = zlib.decompress(base64.b64decode(b64_str)).decode('utf-8')
    if is_print:
        print(json_str)
    return json_str

def get_compressed_model(model_name, compressed_json):
    r""" Get compressed (INT8) models from existing `compressed_json` dict

    Parameters
    ----------
    model_name: str
        String value represents the name of compressed (INT8) model.
    compressed_json : dict
        Dictionary's key represents the name of (INT8) model, and dictionary's value
        represents the compressed json string of (INT8) model.
    """
    b64_str = compressed_json.get(model_name, None)
    if b64_str:
        return decode_b64(b64_str)
    raise ValueError('Model: {} is not found. Available compressed models are:\n{}'.format(model_name, '\n'.join(list(compressed_json.keys()))))
