import argparse

from mxnet import nd, image

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data import ImageNet1kAttr
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, required=True,
                    help='path to the input picture')
opt = parser.parse_args()

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
net = get_model(model_name, pretrained=pretrained)

if not pretrained:
    net.load_parameters(opt.saved_params)
    attrib = ImageNet1kAttr()
    classes = attrib.classes
else:
    classes = net.classes

# Load Images
img = image.imread(opt.input_pic)

# Transform
img = transform_eval(img)
pred = net(img)

topK = 5
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input picture is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.'%
          (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
