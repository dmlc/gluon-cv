"""Pose utils"""
import time
import logging
import datetime

import numpy as np
import torch

from .. import comm
from ..utils import AverageMeter
from ..model_utils import save_model
from ..optimizer import maybe_add_gradient_clipping
from ..eval_utils.coco_eval import COCOEvaluator

logger = logging.getLogger(__name__)

def _detect_anomaly(losses, loss_dict, iteration):
    if not torch.isfinite(losses).all():
        raise FloatingPointError(
            "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                iteration, loss_dict
            )
        )


class DirectposePipeline:
    """Train/eval directpose"""
    def __init__(self, base_iter, max_iter, model, dataloader, optimizer, scheduler,
                 cfg, writer=None):
        logger.info('Configuration:\n' + str(cfg))
        self.base_iter = base_iter
        self.max_iter = max_iter
        self.model = model
        self.dataloader = iter(dataloader)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.writer = writer
        self.iter_timer = AverageMeter()
        val_datasets = cfg.CONFIG.DATA.DATASET.VAL
        self.evaluators = [COCOEvaluator(
            val_dataset, cfg, distributed=cfg.DDP_CONFIG.DISTRIBUTED,
            output_dir=cfg.CONFIG.LOG.EVAL_DIR) for val_dataset in val_datasets]

    def train_step(self):
        cfg = self.cfg
        writer = self.writer
        self.model.train()
        end = time.perf_counter()
        data = next(self.dataloader)
        self.base_iter += 1
        if self.base_iter >= self.max_iter:
            return

        data_time = time.perf_counter() - end

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        _detect_anomaly(losses, loss_dict, self.base_iter)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        batch_time = time.perf_counter() - end
        end = time.perf_counter()
        metrics_dict["batch_time"] = batch_time

        # gather all metrics
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        all_metrics_dict = comm.gather(metrics_dict)
        if self.base_iter % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            eta_str = None
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            if "batch_time" in all_metrics_dict[0]:
                # batch_time among workers can have high variance. The actual latency
                # caused by batch_time is the maximum among workers.
                batch_time = np.max([x.pop("batch_time") for x in all_metrics_dict])
                self.iter_timer.update(batch_time)
                eta = (self.max_iter - self.base_iter) * self.iter_timer.avg
                eta_str = str(datetime.timedelta(seconds=int(eta)))
            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())
            for param in self.optimizer.param_groups:
                lr = param['lr']
            print_string = 'Iter: [{0}/{1}]'.format(
                self.base_iter, self.max_iter)
            if eta_str is not None:
                print_string += f' ETA: {eta_str} '
            print_string += ' data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time, batch_time=batch_time)
            print_string += ' loss: {loss:.5f}'.format(loss=total_losses_reduced)
            print_string += ' lr: {lr:.6f}'.format(lr=lr)
            iteration = self.base_iter
            writer.add_scalar('total_loss', total_losses_reduced, iteration)
            writer.add_scalar('learning_rate', lr, iteration)
            if len(metrics_dict) > 1:
                for km, kv in metrics_dict.items():
                    writer.add_scalar(km, kv, iteration)
                    print_string += f' {km}: {kv:.2f}'
            logger.info(print_string)
        self.scheduler.step()

    def validate(self, val_loader):
        if not self.evaluators:
            self.info('No evaluation data specified, skip...')
            return
        # save model before evaluation so we may recover from bugs
        self.save_model()
        is_training = self.model.training
        self.model.eval()
        for evaluator in self.evaluators:
            evaluator.reset()

        # inference
        num_devices = comm.get_world_size()
        total = len(val_loader)
        logger.info(f"Start evaluation on {total} images.")
        warmup_meter = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        with torch.no_grad():
            for idx, inputs in enumerate(val_loader):
                if idx <= warmup_meter:
                    # no timing during gpu warm up
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                outputs = self.model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                for evaluator in self.evaluators:
                    evaluator.process(inputs, outputs)
                iters_after_start = idx + 1 - warmup_meter * int(idx >= warmup_meter)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= warmup_meter * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    logger.info(f"Inference done {idx + 1}/{total}. {seconds_per_img:.4f} sec/img. ETA={str(eta)}")

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_time_str, total_time / (total - warmup_meter), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - warmup_meter), num_devices
            )
        )
        for evaluator in self.evaluators:
            results = evaluator.evaluate()
            if self.cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                assert isinstance(results, dict)
                for task, res in results.items():
                    # Don't print "AP-category" metrics since they are usually not tracked.
                    important_res = [(k, v) for k, v in res.items() if "-" not in k]
                    logger.info("copypaste: Task: {}".format(task))
                    logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
                    logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
                    for k in important_res:
                        self.writer.add_scalar(':'.join((task, k[0])), k[1], self.base_iter)

        # resume training state if applicable
        self.model.train(is_training)

    def save_model(self):
        """Save checkpoint"""
        if self.cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            save_model(self.model, self.optimizer, self.base_iter, self.cfg)


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
