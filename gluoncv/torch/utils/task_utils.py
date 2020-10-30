# pylint: disable=line-too-long
"""
Utility functions for task
"""
import os
import time
import numpy as np

import torch

from .utils import AverageMeter, accuracy


def train_classification(base_iter, model, dataloader, epoch, criterion,
                         optimizer, cfg, writer=None):
    """Task of training video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, data in enumerate(dataloader):
        base_iter = base_iter + 1

        train_batch = data[0].cuda()
        train_label = data[1].cuda()
        data_time.update(time.time() - end)

        outputs = model(train_batch)
        loss = criterion(outputs, train_label)
        prec1, prec5 = accuracy(outputs.data, train_label, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), train_label.size(0))
        top1.update(prec1.item(), train_label.size(0))
        top5.update(prec5.item(), train_label.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                lr = param['lr']
            print('lr: ', lr)
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)
            iteration = base_iter
            writer.add_scalar('train_loss_iteration', losses.avg, iteration)
            writer.add_scalar('train_top1_acc_iteration', top1.avg, iteration)
            writer.add_scalar('train_top5_acc_iteration', top5.avg, iteration)
            writer.add_scalar('train_batch_size_iteration', train_label.size(0), iteration)
            writer.add_scalar('learning_rate', lr, iteration)
    return base_iter


def validation_classification(model, val_dataloader, epoch, criterion, cfg, writer):
    """Task of validating video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, data in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            val_batch = data[0].cuda()
            val_label = data[1].cuda()
            outputs = model(val_batch)

            loss = criterion(outputs, val_label)
            prec1a, prec5a = accuracy(outputs.data, val_label, topk=(1, 5))

            losses.update(loss.item(), val_batch.size(0))
            top1.update(prec1a.item(), val_batch.size(0))
            top5.update(prec5a.item(), val_batch.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)

        eval_path = cfg.CONFIG.LOG.EVAL_DIR
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        with open(os.path.join(eval_path, "{}.txt".format(cfg.DDP_CONFIG.GPU_WORLD_RANK)), 'w') as f:
            f.write("{} {} {}\n".format(losses.avg, top1.avg, top5.avg))
        torch.distributed.barrier()

        loss_lst, top1_lst, top5_lst = [], [], []
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and writer is not None:
            print("Collecting validation numbers")
            for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE):
                data = open(os.path.join(eval_path, "{}.txt".format(x))).readline().strip().split(" ")
                data = [float(x) for x in data]
                loss_lst.append(data[0])
                top1_lst.append(data[1])
                top5_lst.append(data[2])
            print("Global result:")
            print_string = 'loss: {loss:.5f}'.format(loss=np.mean(loss_lst))
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=np.mean(top1_lst),
                top5_acc=np.mean(top5_lst))
            print(print_string)
            writer.add_scalar('val_loss_epoch', np.mean(loss_lst), epoch)
            writer.add_scalar('val_top1_acc_epoch', np.mean(top1_lst), epoch)
            writer.add_scalar('val_top5_acc_epoch', np.mean(top5_lst), epoch)


def test_classification(model, test_loader, criterion, cfg, file):
    """Task of testing video classification"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    final_result = []

    with torch.no_grad():
        for step, (inputs, labels, ids, chunk_nb, split_nb) in enumerate(test_loader):
            data_time.update(time.time() - end)
            val_batch = inputs.cuda()
            val_label = labels.cuda()
            outputs = model(val_batch)
            loss = criterion(outputs, val_label)

            for i in range(outputs.size(0)):
                string = "{} {} {} {} {}\n".format(ids[i], \
                                                   str(outputs.data[i].cpu().numpy().tolist()), \
                                                   str(int(labels[i].cpu().numpy())), \
                                                   str(int(chunk_nb[i].cpu().numpy())), \
                                                   str(int(split_nb[i].cpu().numpy())))
                final_result.append(string)

            prec1, prec5 = accuracy(outputs.data, val_label, topk=(1, 5))
            losses.update(loss.item(), val_batch.size(0))
            top1.update(prec1.item(), val_batch.size(0))
            top5.update(prec5.item(), val_batch.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 and cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
                print('----Testing----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(0, step + 1, len(test_loader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)
    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(top1.avg, top5.avg))
        for line in final_result:
            f.write(line)
