"""Script for export pre-trained models in GluonCV model zoo."""
from __future__ import print_function
import argparse
import gluoncv as gcv

def parse_args():
    parser = argparse.ArgumentParser("Export model helper.")
    parser.add_argument('--model', '-m', required=True, type=str, help='Name of the model')
    parser.add_argument('--no-preprocess', action='store_true', help='Do not include standard preprocess.')
    args = parser.parse_args()
    return args

args = parse_args()
net = gcv.model_zoo.get_model(args.model, pretrained=True)
gcv.utils.export_block(args.model, net, preprocess=(not args.no_preprocess), layout='HWC')
print('Done...')
