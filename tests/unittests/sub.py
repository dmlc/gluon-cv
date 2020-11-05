import json
import argparse
import os
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='path to the coco image data')
parser.add_argument('--annotation', type=str, required=True, help='path to the annotation')
parser.add_argument('--dest', type=str, default='.', help='path to store the result')
args = parser.parse_args()

data_path = os.path.expanduser(args.data)
annotation_path = os.path.expanduser(args.annotation)
dest = os.path.expanduser(args.dest)
coco = json.load(open(annotation_path))
images = coco['images']

if not os.path.exists(dest):
    os.makedirs(dest)

for image in images:
    image_name = image['file_name']
    image_path = os.path.join(data_path, image_name)
    temp_path = os.path.join(dest, image_name)
    print(f'copying {image_path} to {dest}')
    copyfile(image_path, temp_path)
