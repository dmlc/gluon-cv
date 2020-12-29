"""
Script to compute latency and fps of a model
"""
import os
import argparse
import time

import torch
from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.engine.config import get_cfg_defaults


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FLOPs of a model.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    parser.add_argument('--num-frames', type=int, default=32, help='temporal clip length.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--num-runs', type=int, default=105,
                        help='number of runs to compute average forward timing. default is 105')
    parser.add_argument('--num-warmup-runs', type=int, default=5,
                        help='number of warmup runs to avoid initial slow speed. default is 5')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    model = get_model(cfg)
    model.eval()
    model.cuda()
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, args.num_frames, args.input_size, args.input_size)).cuda()
    print('Model is loaded, start forwarding.')

    with torch.no_grad():
        for i in range(args.num_runs):
            if i == args.num_warmup_runs:
                start_time = time.time()
            pred = model(input_tensor)

    end_time = time.time()
    total_forward = end_time - start_time
    print('Total forward time is %4.2f seconds' % total_forward)

    actual_num_runs = args.num_runs - args.num_warmup_runs
    latency = total_forward / actual_num_runs
    fps = (cfg.CONFIG.DATA.CLIP_LEN * cfg.CONFIG.DATA.FRAME_RATE) * actual_num_runs / total_forward

    print("FPS: ", fps, "; Latency: ", latency)
