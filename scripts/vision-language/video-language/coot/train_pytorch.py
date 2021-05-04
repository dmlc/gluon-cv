import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from tensorboardX import SummaryWriter

from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.data import create_datasets, create_loaders
from gluoncv.torch.utils.model_utils import deploy_model, load_model, save_model
from gluoncv.torch.utils.task_utils import train_coot, validate_coot
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.engine.launch import spawn_workers
from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.utils.lr_policy import ReduceLROnPlateau, ReduceLROnPlateauWarmup
from gluoncv.torch.utils.optimizer import RAdam
from gluoncv.torch.utils.loss import MaxMarginRankingLoss, CycleConsistencyCootLoss
from gluoncv.torch.utils.coot_utils import get_logger, close_logger, compare_metrics, create_dataloader_path


def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.freeze()
    logger = get_logger(tb_logdir, "trainer", log_file=True)

    # create model
    model = get_model(cfg)
    model = deploy_model(model, cfg)

    # create dataset and dataloader
    data_path_dict = create_dataloader_path(
        cfg.CONFIG.COOT_DATA.DATA_PATH,
        cfg.CONFIG.COOT_DATA.DATASET_NAME,
        video_feature_name=cfg.CONFIG.COOT_DATA.FEATURE)

    train_set, val_set = create_datasets(data_path_dict, cfg,
                                         cfg.CONFIG.COOT_DATA.VIDEO_PRELOAD,
                                         cfg.CONFIG.COOT_DATA.TEXT_PRELOAD)
    train_loader, val_loader = create_loaders(train_set, val_set,
                                              cfg.CONFIG.TRAIN.BATCH_SIZE,
                                              cfg.CONFIG.DATA.NUM_WORKERS)
    optimizer = RAdam(model.get_params(),
                      lr=cfg.CONFIG.TRAIN.LR,
                      betas=(cfg.CONFIG.TRAIN.MOMENTUM,
                             cfg.CONFIG.TRAIN.ADAM_BETA2),
                      eps=cfg.CONFIG.TRAIN.ADAM_EPS,
                      weight_decay=cfg.CONFIG.TRAIN.W_DECAY)

    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, optimizer, cfg, load_fc=True)

    if cfg.CONFIG.TRAIN.LR_POLICY == 'Step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.CONFIG.TRAIN.LR_MILESTONE,
            gamma=cfg.CONFIG.TRAIN.STEP)

    elif cfg.CONFIG.TRAIN.LR_POLICY == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.CONFIG.TRAIN.EPOCH_NUM - cfg.CONFIG.TRAIN.WARMUP_EPOCHS,
            eta_min=0,
            last_epoch=cfg.CONFIG.TRAIN.RESUME_EPOCH)

    elif cfg.CONFIG.TRAIN.LR_POLICY == 'LR_Warmup':
        scheduler = ReduceLROnPlateauWarmup(optimizer,
                                            cfg.CONFIG.TRAIN.WARMUP_EPOCHS,
                                            mode="max",
                                            patience=cfg.CONFIG.TRAIN.PATIENCE,
                                            cooldown=cfg.CONFIG.TRAIN.COOLDOWN)

    else:
        print(
            'Learning rate schedule %s is not supported yet. Please use Step or Cosine.'
        )

    criterion_cycleconsistency = CycleConsistencyCootLoss(num_samples=1,
                                                          use_cuda=True)
    criterion_alignment = MaxMarginRankingLoss(use_cuda=True)

    base_iter = 0
    det_best_field_best = 0
    for epoch in range(cfg.CONFIG.TRAIN.EPOCH_NUM):

        ## ======== Training step ===============
        base_iter = train_coot(cfg, base_iter, model, train_loader, epoch,
                               criterion_alignment, criterion_cycleconsistency,
                               optimizer, writer, logger)

        ## ======= Validation step ================
        if epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1:
            vid_metrics, clip_metrics = validate_coot(cfg,
                model, val_loader, epoch, criterion_alignment,
                criterion_cycleconsistency, writer, logger, True)

        # Check if the performance of model is improving
        logger.info("---------- Validating epoch {} ----------".format(epoch))
        c2s_res, s2c_res, clip_best_at_1 = None, None, None
        if clip_metrics is not None:
            c2s_res, s2c_res, clip_best_at_1 = clip_metrics

        # find field which determines is_best
        det_best_field_current = clip_best_at_1

        # check if best
        is_best = compare_metrics(det_best_field_current, det_best_field_best)
        if is_best:
            det_best_field_best = det_best_field_current
            best_epoch = epoch

        # step lr scheduler
        scheduler.step_rop(det_best_field_current, True)
        logger.info(f"ROP: model improved: {is_best}, "
                    f"value {det_best_field_current:.3f},"
                    f"new LR: {optimizer.param_groups[0]['lr']:5.3e}")

        if epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0:
            if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 or cfg.DDP_CONFIG.DISTRIBUTED == False:
                model.save_model(optimizer, epoch, cfg)

        # check if model did not improve for too long
        term_after = 15
        if epoch - best_epoch > term_after:
            logger.info(f"NO improvements for {term_after} epochs (current "
                        f"{epoch} best {best_epoch}) STOP training.")
            break

    if writer is not None:
        writer.close()
    if logger is not None:
        close_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Coot model.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    args = parser.parse_args()
    cfg = get_cfg_defaults(name='coot')
    cfg.merge_from_file(args.config_file)
    main_worker(cfg)
