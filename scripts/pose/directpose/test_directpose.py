import os
from gluoncv.torch import model_zoo
from gluoncv.torch.engine.config import get_cfg_defaults
import torch
import torchvision.transforms as T
from PIL import Image
import tvm
from tvm import relay
from tvm.contrib.download import download
from torchviz import make_dot, make_dot_from_trace

# def debug_trace(kpt_bases):
#     stride = 1
#     # a = kpt_bases[:, [1, 0], :, :].contiguous() * stride + stride // 2
#     a = torch.cat((kpt_bases[:, 1:2], kpt_bases[:, 0:1]), dim=1) * stride + stride // 2
#     print(a.shape)
#     return a

def export_onnx(model, input):
    torch.onnx.export(model, input, 'model.onnx')

def get_image(img_name='street_small.jpg', img_url=None):
    def get_single_image_input(img_url, save_dir='/tmp'):
        img_name = img_url.split("/")[-1]
        img_path = os.path.join(save_dir, img_name)
        download(img_url, img_path)
        orig_img = Image.open(img_path).convert("RGB")
        img = orig_img.resize((800, 1280), Image.LANCZOS)
        return img, orig_img, img_path

    def get_transforms():
        tforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return tforms
    
    if img_url is None:
        img_url = f"https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/{img_name}"
    img, orig_img, img_path = get_single_image_input(img_url)

    tforms = get_transforms()
    input_data = tforms(img).unsqueeze(0).float()
    return input_data, orig_img


if __name__ == '__main__':
    device = torch.device('cuda')
    cfg = get_cfg_defaults()
    # cfg.merge_from_file('./configurations/ms_dla_34_4x_syncbn.yaml')
    # net = model_zoo.dla34_fpn_directpose(cfg).to(device).eval()
    # model = torch.load('model_final.pth')['model']
    cfg.merge_from_file('./configurations/ms_aa_resnet50_4x_syncbn.yaml')
    net = model_zoo.resnet_lpf_fpn_directpose(cfg).to(device).eval()
    model = torch.load('model_final_resnet.pth')['model']
    net.load_state_dict(model, strict=False)
    # images = torch.zeros(1, 3, 512, 512).cuda()
    images, orig_image = get_image()
    y = net(images.to(device))
    print(y)
    raise
    # with torch.no_grad():
    #     scripted_model = torch.jit.trace(debug_trace, torch.zeros(1, 2, 48, 48))
    # torch._C._jit_pass_inline(scripted_model.graph)
    # print(scripted_model.graph)
    # export_onnx(net, images)
    # raise
    with torch.no_grad():
        scripted_model = torch.jit.trace(net.forward, images.to(device)).eval()

    torch._C._jit_pass_inline(scripted_model.graph)
    print(scripted_model.graph)
    input_name = "input0"
    shape_list = [(input_name, images.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    target = "cuda"
    target_host = "llvm"
    ctx = tvm.gpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    export_graph, export_lib, export_params = lib
    export_lib.export_library('compiled.so')
    with open('compiled.json', 'w') as f:
        f.write(export_graph)
    with open('compiled.params', 'wb') as f:
        f.write(relay.save_param_dict(export_params))
    print('export complete')

    from tvm.contrib import graph_runtime

    dtype = "float32"
    m = graph_runtime.GraphModule(lib["default"](ctx))
    # Set inputs
    m.set_input(input_name, tvm.nd.array(images.cpu().detach().numpy().astype(dtype)))
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
    print(tvm_output)
