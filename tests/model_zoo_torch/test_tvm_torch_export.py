import tvm
from tvm import relay
from tvm.contrib import graph_executor

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from torchvision import transforms
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch import model_zoo

def _test_tvm_export(model, img, target="llvm"):
  model = model.eval()

  # We grab the TorchScripted model via tracing
  input_shape = [1, 3, 224, 224]
  input_data = torch.randn(input_shape)
  scripted_model = torch.jit.trace(model, input_data).eval()
  input_name = "input0"
  shape_list = [(input_name, img.shape)]
  mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
  if target == "llvm":
    ctx = tvm.cpu(0)
  elif target == "cuda":
    ctx = tvm.gpu(0)
  else:
    raise ValueError(target)
  target_host = "llvm"

  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target, target_host=target_host, params=params)

  dtype = "float32"
  m = graph_executor.GraphModule(lib["default"](ctx))
  # Set inputs
  m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
  # Execute
  m.run()
  # Get outputs
  tvm_output = m.get_output(0)

# def test_model_zoo_tvm_export():
#   from tvm.contrib.download import download_testdata
#   img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
#   img_path = download_testdata(img_url, "cat.png", module="data")
#   img = Image.open(img_path).resize((224, 224))
#   my_preprocess = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
#   )
#   img = my_preprocess(img)
#   img = np.expand_dims(img, 0)
#
#   model = getattr(torchvision.models, "resnet18")(pretrained=True)
#   _test_tvm_export(model, img)

def test_model_zoo_tvm_export_directpose():
    from gluoncv.torch.utils.tvm_utils.nms import nms
    from tvm.contrib.download import download_testdata
    def get_image(img_name='street_small.jpg', img_url=None, img_width=1280, img_height=736):
        def get_single_image_input(img_url):
            img_name = img_url.split("/")[-1]
            img_path = download_testdata(img_url, img_name, module="data")
            orig_img = Image.open(img_path)
            orig_img = orig_img.resize((img_width, img_height), Image.LANCZOS)
            img = np.array(orig_img)[:, :, (0, 1, 2)].astype('uint8')
            return img, orig_img, img_path

        def get_transforms():
            tforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            return tforms

        if img_url is None:
            img_url = f"https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/{img_name}"
        img, orig_img, img_path = get_single_image_input(img_url)

        tforms = get_transforms()
        input_data = tforms(img).unsqueeze(0)
        return input_data, orig_img

    images, orig_img = get_image()
    model_list = ['directpose_resnet50_lpf_fpn_coco']

    cfg = get_cfg_defaults(name='directpose')
    torch.set_grad_enabled(False)
    use_cuda = True
    if torch.cuda.device_count() == 0:
        use_cuda = False
    if use_cuda:
        device = torch.device('cuda')
        target = "cuda"
        ctx = tvm.gpu(0)
    else:
        device = torch.device('cpu')
        target = "llvm"
        ctx = tvm.cpu()
    for model in model_list:
        cfg.CONFIG.MODEL.PRETRAINED = True
        cfg.CONFIG.MODEL.NAME = model
        cfg.CONFIG.MODEL.TVM_MODE = True
        net = model_zoo.get_model(cfg).to(device).eval()

        # y = net(images.to(device))
        with torch.no_grad():
            scripted_model = torch.jit.trace(net.forward, images.to(device)).eval()

        input_name = "input0"
        shape_list = [(input_name, images.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, {"torchvision::nms": nms})
        target_host = "llvm"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)

if __name__ == '__main__':
    import nose

    nose.runmodule()
