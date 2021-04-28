import os
import argparse
import numpy as np
from gluoncv.torch import model_zoo
from gluoncv.torch.engine.config import get_cfg_defaults
import torch
import torchvision.transforms as T
from PIL import Image
import tvm
from tvm import relay
from tvm.contrib.download import download

from tvm.relay.frontend.pytorch import _op, AttrCvt, get_relay_op
def nms(inputs, input_types):
    """A static NMS hack for torchvision.nms, it requires the nms to be the last layer of the network"""
    boxes = inputs[0]
    scores = inputs[1]
    iou_threshold = inputs[2]
    score_index = 0
    coord_start=1
    # PyTorch NMS doesn't have parameter top_k and max_output_size
    top_k = max_out_size = -1

    # TVM NMS assumes score > 0
    scores = scores - _op.min(scores) + _op.const(1.0)

    # a tricky to preserve the original index of bounding boxes, as the nms op will re-arrange the array
    indices = _op.transform.arange(_op.squeeze(_op.shape_of(scores)), dtype="int32")
    indices = _op.cast(indices, dtype="float32")
    indices = _op.expand_dims(indices, -1, 1)

    # More efficient way to run NMS for torchvision.nms
    scores = AttrCvt(op_name="expand_dims", extras={"axis": -1, "num_newaxis": 1})([scores], {})

    data = _op.concatenate([scores, boxes, indices], -1)
    data = _op.expand_dims(data, 0, 1)
    valid_ret = _op.vision.get_valid_counts(
        data, score_threshold=1.05, id_index=5, score_index=score_index
    )

    nms_ret = get_relay_op("non_max_suppression")(
        data=valid_ret[1],
        valid_count=valid_ret[0],
        indices=valid_ret[2],
        max_output_size=max_out_size,
        iou_threshold=iou_threshold,
        force_suppress=True,
        top_k=top_k,
        coord_start=1,
        score_index=score_index,
        id_index=5,
        return_indices=False,
        invalid_to_bottom=False,
    )

    nms_ret = relay.squeeze(nms_ret, axis=[0])
    return nms_ret

def get_image(img_name='street_small.jpg', img_url=None, img_width=1280, img_height=736):
    def get_single_image_input(img_url, save_dir='/tmp'):
        img_name = img_url.split("/")[-1]
        img_path = os.path.join(save_dir, img_name)
        download(img_url, img_path)
        orig_img = Image.open(img_path)
        # img = orig_img.resize((736, 1280), Image.LANCZOS)
        img = orig_img.resize((img_width, img_height), Image.LANCZOS)
        img = np.array(img)[:, :, (2, 1, 0)]
        return img, orig_img, img_path

    def get_transforms():
        tforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.229, 0.224, 0.225])])
        return tforms

    if img_url is None:
        img_url = f"https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/{img_name}"
    img, orig_img, img_path = get_single_image_input(img_url)

    tforms = get_transforms()
    input_data = tforms(img).unsqueeze(0)
    return input_data, orig_img

def parse_args():
    parser = argparse.ArgumentParser(description='TVM exporter for directpose')
    parser.add_argument('--model-prefix', type=str, default='./compiled')
    parser.add_argument('--use-gpu', type=bool, default=True)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=736)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--export-dir", type=str, default='./')
    parser.add_argument("--model-name", type=str, default='', help="The pre-trained model name")
    parser.add_argument("--config-file", type=str, default='./configurations/ms_aa_resnet50_4x_syncbn.yaml',
                        help="The config file for custom model, required if --model-name is not specified")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.set_grad_enabled(False)
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cfg = get_cfg_defaults(name='directpose')
    if args.model_name:
        cfg.CONFIG.MODEL.PRETRAINED = True
        cfg.CONFIG.MODEL.NAME = args.model_name
        net = model_zoo.get_model(cfg)
    else:
        assert os.path.isfile(args.config_file)
        assert os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        cfg.merge_from_file(args.config_file)
        net = model_zoo.directpose_resnet_lpf_fpn(cfg).to(device).eval()
        load_model = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        net.load_state_dict(load_model, strict=False)
    images, orig_image = get_image(
        'soccer.png', img_url='https://github.com/dmlc/web-data/blob/master/gluoncv/pose/soccer.png?raw=true',
        img_width=args.image_width, img_height=args.image_height)
    y = net(images.to(device))
    with torch.no_grad():
        scripted_model = torch.jit.trace(net.forward, images.to(device)).eval()

    input_name = "input0"
    shape_list = [(input_name, images.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, {"torchvision::nms": nms})

    if args.verbose:
        print(mod)

    if args.use_gpu:
        target = "cuda"
        ctx = tvm.gpu(0)
    else:
        target = "llvm"
        ctx = tvm.cpu()
    target_host = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    export_graph, export_lib, export_params = lib
    export_lib.export_library('compiled.so')
    with open('compiled.json', 'w') as f:
        f.write(export_graph)
    with open('compiled.params', 'wb') as f:
        f.write(relay.save_param_dict(export_params))
    print('export complete')

    if args.verbose:
        print('Running sanity check in tvm runtime')
        from tvm.contrib import graph_runtime
        dtype = "float32"
        m = graph_runtime.GraphModule(lib["default"](ctx))
        # Set inputs
        m.set_input(input_name, tvm.nd.array(images.cpu().detach().numpy().astype(dtype)))
        tvm_result = m.run()
