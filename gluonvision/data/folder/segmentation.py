import os
import math
import random
from PIL import Image, ImageOps, ImageFilter

import mxnet as mx
from mxnet import gluon

class Segmentation(gluon.data.Dataset):
    def __init__(self, root, transform=None):
    def __init__(self, root, train='test', transform=None):
        self.root = root
        self.transform = transform
        self.images = _get_folder_images(root)
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

def _get_folder_images(img_folder):
    img_paths = []  
    for filename in os.listdir(img_folder):
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            img_paths.append(imgpath)
    return img_paths
