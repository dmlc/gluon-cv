import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from tensorboardX import SummaryWriter

from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.model_zoo.pose import directpose_resnet_lpf_fpn
from gluoncv.torch.data.pose import build_pose_train_loader, build_pose_test_loader
from gluoncv.torch.utils.model_utils import deploy_model, load_model, save_model
from gluoncv.torch.utils.task_utils import DirectposePipeline, build_pose_optimizer
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.engine.launch import spawn_workers
from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.utils.lr_policy import build_lr_scheduler


def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.freeze()

    # create model
    # model = get_model(cfg)
    model = directpose_resnet_lpf_fpn(cfg)
    model = deploy_model(model, cfg)

    # create dataset and dataloader
    # train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_pose_train_loader(cfg)
    train_loader = build_pose_train_loader(cfg)
    val_loader = build_pose_test_loader(cfg, cfg.CONFIG.DATA.DATASET.VAL[0])
    optimizer = build_pose_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # if cfg.CONFIG.MODEL.LOAD:
    #     model, _ = load_model(model, optimizer, cfg, load_fc=True)
    # criterion = nn.CrossEntropyLoss().cuda()

    pipeline = DirectposePipeline(0, cfg.CONFIG.TRAIN.ITER_NUM, model, train_loader, optimizer, scheduler, cfg, writer=writer)
    while pipeline.base_iter < pipeline.max_iter:
        pipeline.train_step()

        if pipeline.base_iter % cfg.CONFIG.VAL.EVAL_PERIOD == 0 or pipeline.base_iter == pipeline.max_iter:
            pipeline.validate(val_loader)

        if pipeline.base_iter % cfg.CONFIG.TRAIN.CHECKPOINT_PERIOD == 0:
            if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 or cfg.DDP_CONFIG.DISTRIBUTED == False:
                pipeline.save_model()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition models.')
    parser.add_argument('--config-file', type=str, help='path to config file.', required=True)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg_defaults(name='directpose')
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.CONFIG.MODEL.DIRECTPOSE.ENABLE_HM_BRANCH = True  # enable HM_BRANCH in training
    spawn_workers(main_worker, cfg)
