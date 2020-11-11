"""
Script to compute FLOPs of a model
"""
import os
import argparse

import torch
from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.engine.config import get_cfg_defaults

from thop import profile, clever_format


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FLOPs of a model.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    parser.add_argument('--num-frames', type=int, default=32, help='temporal clip length.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    model = get_model(cfg)
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, args.num_frames, args.input_size, args.input_size))

    macs, params = profile(model, inputs=(input_tensor,))
    macs, params = clever_format([macs, params], "%.3f")
    print("FLOPs: ", macs, "; #params: ", params)
