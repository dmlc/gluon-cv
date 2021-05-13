import os
import argparse
from gluoncv.torch import model_zoo
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.utils.model_utils import download
from gluoncv.torch.utils.visualizer import ColorMode, Visualizer
from gluoncv.torch.data.registry.catalog import MetadataCatalog
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description='TVM exporter for directpose')
    parser.add_argument('--model-prefix', type=str, default='./compiled')
    parser.add_argument('--use-gpu', type=int, default=1)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=736)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--visualize", type=int, default=1, help="enable visualizer")
    parser.add_argument("--save-output", type=str, default='visualize_output.png', help='Save visualize result to image')
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--export-dir", type=str, default='./')
    parser.add_argument("--model-name", type=str, default='directpose_resnet50_lpf_fpn_coco', help="The pre-trained model name")
    parser.add_argument("--config-file", type=str, default='./configurations/ms_aa_resnet50_4x_syncbn.yaml',
                        help="The config file for custom model, required if --model-name is not specified")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def get_image(img_name='street_small.jpg', img_url=None, img_width=1280, img_height=736):
    def get_single_image_input(img_url, save_dir='/tmp'):
        img_name = img_url.split("/")[-1]
        img_path = os.path.join(save_dir, img_name)
        download(img_url, img_path)
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

if __name__ == '__main__':
    args = parse_args()
    torch.set_grad_enabled(False)
    cfg = get_cfg_defaults(name='directpose')
    if args.use_gpu:
        print('Using GPU...')
        device = torch.device('cuda')
    else:
        print('Using CPU...')
        device = torch.device('cpu')
    if args.model_name:
        cfg.CONFIG.MODEL.PRETRAINED = True
        cfg.CONFIG.MODEL.NAME = args.model_name
        net = model_zoo.get_model(cfg).to(device).eval()
    else:
        assert os.path.isfile(args.config_file)
        assert os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        cfg.merge_from_file(args.config_file)
        net = model_zoo.directpose_resnet_lpf_fpn(cfg).to(device).eval()
        load_model = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH, map_location=torch.device('cpu'))
        net.load_state_dict(load_model, strict=False)
    images, orig_image = get_image(
        'soccer.png', img_url='https://github.com/dmlc/web-data/blob/master/gluoncv/pose/soccer.png?raw=true',
        img_width=args.image_width, img_height=args.image_height)
    with torch.no_grad():
        predictions = net(images.to(device))[0]
    if args.score_threshold > 0:
        print(f"Applying score threshold: {args.score_threshold} to detected instances...")
        valid_indices = predictions.scores > args.score_threshold
        predictions = predictions[valid_indices]
    print(f"Detected {list(predictions.scores.size())} instances with scores: {predictions.scores}")
    # visualize
    if args.visualize or args.save_output:
        print("Visualizing results...")
        metadata = MetadataCatalog.get(
            cfg.CONFIG.DATA.DATASET.VAL[0] if len(cfg.CONFIG.DATA.DATASET.VAL) else "__unused"
        )
        visualizer = Visualizer(orig_image, metadata, instance_mode=ColorMode.IMAGE)
        instances = predictions.to(torch.device("cpu"))
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        if args.save_output:
            vis_output.save(args.save_output)
        if args.visualize:
            raise
            plt.imshow(vis_output.get_image())
            plt.show()
