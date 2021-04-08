"""Pose utils"""
import time
import logging

import numpy as np
import torch

from .. import comm
from ..optimizer import maybe_add_gradient_clipping

logger = logging.getLogger(__name__)

def _detect_anomaly(losses, loss_dict, iteration):
    if not torch.isfinite(losses).all():
        raise FloatingPointError(
            "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                iteration, loss_dict
            )
        )

def train_directpose(base_iter,
               model,
               dataloader,
               epoch,
               optimizer,
               cfg,
               writer=None):
    "training pipeline for directpose"

    model.train()
    end = time.perf_counter()
    for step, data in enumerate(dataloader):
        base_iter = base_iter + 1
        data_time = time.perf_counter() - end

        loss_dict = model(data)
        losses = sum(loss_dict.values())
        _detect_anomaly(losses, loss_dict, base_iter)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.perf_counter() - end
        end = time.perf_counter()
        metrics_dict["batch_time"] = batch_timef

        # gather all metrics
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        all_metrics_dict = comm.gather(metrics_dict)
        if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            if "batch_time" in all_metrics_dict[0]:
                # batch_time among workers can have high variance. The actual latency
                # caused by batch_time is the maximum among workers.
                batch_time = np.max([x.pop("batch_time") for x in all_metrics_dict])
            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())
            for param in optimizer.param_groups:
                lr = param['lr']
            print_string = 'Epoch: [{0}][{1}]'.format(
                epoch, step + 1)
            print_string += ' data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time, batch_time=batch_time)
            print_string += ' loss: {loss:.5f}'.format(loss=total_losses_reduced)
            iteration = base_iter
            writer.add_scalar('total_loss', total_losses_reduced, iteration)
            writer.add_scalar('learning_rate', lr, iteration)
            if len(metrics_dict) > 1:
                for km, kv in metrics_dict.items():
                    writer.add_scalar(km, kv, iteration)
                    print_string += f' {km}: {kv:.2f}'
            logger.info(print_string)
    return base_iter

def validate_directpose():
    pass

def build_pose_optimizer(cfg, model) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    override: Set[torch.nn.parameter.Parameter] = set()

    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.CONFIG.TRAIN.LR
            weight_decay = cfg.CONFIG.TRAIN.W_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.CONFIG.TRAIN.W_DECAY_NORM
            elif key == "bias":
                lr = cfg.CONFIG.TRAIN.LR
                weight_decay = cfg.CONFIG.TRAIN.W_DECAY
            if value in override:
                raise NotImplementedError('KPS_GRAD_MULT not found')

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(
        params, cfg.CONFIG.TRAIN.LR, momentum=cfg.CONFIG.TRAIN.MOMENTUM)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer
