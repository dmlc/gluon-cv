from gluoncv.torch import model_zoo
from gluoncv.torch.engine.config import get_cfg_defaults
import torch
import tvm
from tvm import relay

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file('./configurations/ms_dla_34_4x_syncbn.yaml')
    net = model_zoo.dla34_fpn_directpose(cfg).cuda().eval()
    model = torch.load('model_final.pth')['model']
    net.load_state_dict(model, strict=False)
    images = torch.empty(1, 3, 512, 512).cuda()
    y = net(images)
    with torch.no_grad():
        scripted_model = torch.jit.trace(net.forward, images).eval()

    torch._C._jit_pass_inline(scripted_model.graph)
    print(scripted_model.graph)
    input_name = "input0"
    shape_list = [(input_name, images.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    target = "llvm"
    target_host = "llvm"
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
