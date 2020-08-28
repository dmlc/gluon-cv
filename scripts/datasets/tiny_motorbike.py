"""Prepare PASCAL VOC tiny motorbike datasets"""
import os
import autogluon as ag


if __name__ == '__main__':
    root = '~/.mxnet/datasets/'
    filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip', path=root)
    filename = ag.unzip(filename_zip, root=root)
    data_root = os.path.join(root, filename)
    print("dataset saved to: {}".format(data_root))