import mxnet.ndarray as nd
from model import SRGenerator
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import cv2
from mxnet import image
import numpy

ctx = mx.gpu(7)
# dommy_img = nd.random.uniform(0,1,(1,3,96,96))
netG = SRGenerator()
# dommy_out = netG(dommy_img)
netG.load_params('samples/netG_epoch_19900.pth')
netG.collect_params().reset_ctx(ctx)

hr_img_list = ['../datasets/DIV2K_valid_HR/' + str(i).zfill(4)+'.png' for i in range(801,901)]
lr_img_list = ['../datasets/DIV2K_valid_LR_bicubic/X4/'  + str(i).zfill(4)+'x4.png' for i in range(801,901)]
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
for i,lr_img_path in enumerate(lr_img_list):
    lr_img = image.imread(lr_img_path)
    lr_img_in = transform_fn(lr_img).expand_dims(0).as_in_context(ctx)
    hr_img_gen = netG(lr_img_in)
    hr_img_gen = (hr_img_gen[0].transpose([1,2,0]) + 1) / 2 * 255
    hr_img = image.imread(hr_img_list[i])
    lr_img_up = image.imresize(lr_img,lr_img.shape[1]*4,lr_img.shape[0]*4,interp=3)
    all = nd.concatenate([lr_img_up,hr_img_gen.astype(lr_img_up.dtype()),hr_img],axis=0)
    cv2.imwrite('../datasets/valid_gen/'+str(i).zfill(4)+'.png',all.asnumpy())
