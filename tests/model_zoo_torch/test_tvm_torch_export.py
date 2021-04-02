import tvm
from tvm import relay
from tvm.contrib import graph_executor

from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms

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
  
def test_model_zoo_tvm_export():
  from tvm.contrib.download import download_testdata
  img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
  img_path = download_testdata(img_url, "cat.png", module="data")
  img = Image.open(img_path).resize((224, 224))
  my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
  )
  img = my_preprocess(img)
  img = np.expand_dims(img, 0)

  model = getattr(torchvision.models, "resnet18")(pretrained=True)
  _test_tvm_export(model, img)
