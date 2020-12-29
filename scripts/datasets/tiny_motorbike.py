"""Prepare PASCAL VOC tiny motorbike datasets"""
import os
import autogluon as ag


if __name__ == '__main__':
    root = os.path.expanduser('~/.mxnet/datasets/')
    if not os.path.exists(root):
        os.makedirs(root)

    filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip', path=root)
    filename = ag.unzip(filename_zip, root=root)
    data_root = os.path.join(root, filename)
    os.remove(filename_zip)

    print("dataset saved to: {}".format(data_root))