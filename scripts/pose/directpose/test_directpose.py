from gluoncv.torch import model_zoo
from gluoncv.torch.engine.config import get_cfg_defaults
import torch

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file('./configurations/ms_dla_34_4x_syncbn.yaml')
    net = model_zoo.dla34_fpn_directpose(cfg).cuda().eval()
    images = torch.empty(1, 3, 512, 512).cuda()
    y = net(images)
    print(y)
    module = torch.jit.trace(net.forward, images)
    
