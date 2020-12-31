# pylint: disable=line-too-long
"""
Utility functions for task
"""
import os
import time
from timeit import default_timer as timer
import numpy as np

import torch
from torch.nn import functional as F

from gluoncv.torch.utils.coot_utils import compute_constrastive_loss, compute_cmc_loss
from . import coot_utils
from .coot_utils import unpack_data
from .utils import AverageMeter, accuracy


def train_classification(base_iter,
                         model,
                         dataloader,
                         epoch,
                         criterion,
                         optimizer,
                         cfg,
                         writer=None):
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
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(
                epoch, step + 1, len(dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val, batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg, top5_acc=top5.avg)
            print(print_string)
            iteration = base_iter
            writer.add_scalar('train_loss_iteration', losses.avg, iteration)
            writer.add_scalar('train_top1_acc_iteration', top1.avg, iteration)
            writer.add_scalar('train_top5_acc_iteration', top5.avg, iteration)
            writer.add_scalar('train_batch_size_iteration',
                              train_label.size(0), iteration)
            writer.add_scalar('learning_rate', lr, iteration)
    return base_iter


def validation_classification(model, val_dataloader, epoch, criterion, cfg,
                              writer):
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
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(
                    epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val, batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg, top5_acc=top5.avg)
                print(print_string)

        eval_path = cfg.CONFIG.LOG.EVAL_DIR
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        with open(
                os.path.join(eval_path,
                             "{}.txt".format(cfg.DDP_CONFIG.GPU_WORLD_RANK)),
                'w') as f:
            f.write("{} {} {}\n".format(losses.avg, top1.avg, top5.avg))
        torch.distributed.barrier()

        loss_lst, top1_lst, top5_lst = [], [], []
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and writer is not None:
            print("Collecting validation numbers")
            for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE):
                data = open(os.path.join(
                    eval_path,
                    "{}.txt".format(x))).readline().strip().split(" ")
                data = [float(x) for x in data]
                loss_lst.append(data[0])
                top1_lst.append(data[1])
                top5_lst.append(data[2])
            print("Global result:")
            print_string = 'loss: {loss:.5f}'.format(loss=np.mean(loss_lst))
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=np.mean(top1_lst), top5_acc=np.mean(top5_lst))
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
        for step, (inputs, labels, ids, chunk_nb,
                   split_nb) in enumerate(test_loader):
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
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(
                    0, step + 1, len(test_loader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val, batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg, top5_acc=top5.avg)
                print(print_string)
    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(top1.avg, top5.avg))
        for line in final_result:
            f.write(line)


def train_coot(config, base_iter,
               model,
               dataloader,
               epoch,
               constrastive_loss,
               cmc_loss,
               optimizer,
               writer,
               logger,
               use_cuda=True):
    """Train COOT model

    Args:
        base_iter: training iteration counter
        model: COOT model
        dataloader
        epoch: current epoch number
        constrastive_loss: MaxMargingRanking loss
        cmc_loss: Cross-modal cycle-consistensy loss
        optimizer
        writer: tensorboard writer
        logger
        use_cuda (bool): use GPU

    Returns:
        base_itr: iteration counter after current epoch
    """

    max_step = len(dataloader)
    timer_start_train = timer()
    model.train()
    # train one epoch
    logger.info("---------- Training epoch {} ----------".format(epoch))
    for step, data_dict in enumerate(dataloader):
        base_iter = base_iter + 1
        (_, vid_frames, vid_frames_mask, vid_frames_len, par_cap_vectors,
         par_cap_mask, par_cap_len, clip_num, clip_frames, clip_frames_len,
         clip_frames_mask, sent_num, sent_cap_vectors, sent_cap_mask,
         sent_cap_len) = unpack_data(data_dict, use_cuda)

        # forward pass
        (vid_emb, clip_emb, vid_context, clip_emb_reshape, clip_emb_mask,
         clip_emb_lens) = model.encode_video(vid_frames, vid_frames_mask,
                                             vid_frames_len, clip_num,
                                             clip_frames, clip_frames_len,
                                             clip_frames_mask)
        (par_emb, sent_emb, par_context, sent_emb_reshape, sent_emb_mask,
         sent_emb_lens) = model.encode_paragraph(par_cap_vectors, par_cap_mask,
                                                 par_cap_len, sent_num,
                                                 sent_cap_vectors,
                                                 sent_cap_mask, sent_cap_len)
        loss = compute_constrastive_loss(config, constrastive_loss, vid_emb, par_emb,
                                         clip_emb, sent_emb, vid_context,
                                         par_context)
        loss += compute_cmc_loss(cmc_loss, config.CONFIG.TRAIN.LOSS_CYCLE_CONS_W, clip_emb_reshape,
                                 clip_emb_mask, clip_emb_lens,
                                 sent_emb_reshape, sent_emb_mask,
                                 sent_emb_lens)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        if step % 10 == 0:
            el_time = (timer() - timer_start_train) / 60
            l_ms = len(str(max_step))
            str_step = ("{:" + str(l_ms) + "d}").format(step)
            print_string = (
                f"E{epoch}[{str_step}/{max_step}] T {el_time:.3f}m "
                f"LR {optimizer.param_groups[0]['lr']:5.3e} "
                f"L {loss:.4f}")
            logger.info(print_string)

    time_total = timer() - timer_start_train
    logger.info("Training epoch {} took {:.3f}s ".format(epoch, time_total))

    writer.add_scalar('train_loss_iteration', loss, base_iter)
    writer.add_scalar('train_batch_size_iteration', vid_frames.size(0),
                      base_iter)
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                      base_iter)

    return base_iter


def validate_coot(config, model,
                  val_loader,
                  epoch,
                  constrastive_loss,
                  cmc_loss,
                  writer,
                  logger,
                  use_cuda=True):
    """Validate COOT model

    Args:
        model: COOT model
        dataloader
        epoch: current epoch number
        constrastive_loss: MaxMargingRanking loss
        cmc_loss: Cross-modal cycle-consistensy loss
        writer: tensorboard writer
        logger
        use_cuda (bool): use GPU

    Returns:
        Retrieval performance
    """

    model.eval()
    max_step = len(val_loader)

    # collect embeddings
    vid_emb_list = []
    par_emb_list = []
    clip_emb_list = []
    sent_emb_list = []
    for step, data_dict in enumerate(val_loader):

        (vid_id, vid_frames, vid_frames_mask, vid_frames_len, par_cap_vectors,
         par_cap_mask, par_cap_len, clip_num, clip_frames, clip_frames_len,
         clip_frames_mask, sent_num, sent_cap_vectors, sent_cap_mask,
         sent_cap_len) = unpack_data(data_dict, use_cuda)
        if step == 0:
            print(f"ids {vid_id[:4]}...")
        # forward pass
        (vid_emb, clip_emb, vid_context, clip_emb_reshape, clip_emb_mask,
         clip_emb_lens) = model.encode_video(vid_frames, vid_frames_mask,
                                             vid_frames_len, clip_num,
                                             clip_frames, clip_frames_len,
                                             clip_frames_mask)
        (par_emb, sent_emb, par_context, sent_emb_reshape, sent_emb_mask,
         sent_emb_lens) = model.encode_paragraph(par_cap_vectors, par_cap_mask,
                                                 par_cap_len, sent_num,
                                                 sent_cap_vectors,
                                                 sent_cap_mask, sent_cap_len)
        loss = compute_constrastive_loss(config, constrastive_loss, vid_emb, par_emb,
                                         clip_emb, sent_emb, vid_context,
                                         par_context)
        loss += compute_cmc_loss(cmc_loss, config.CONFIG.TRAIN.LOSS_CYCLE_CONS_W, clip_emb_reshape,
                                 clip_emb_mask, clip_emb_lens,
                                 sent_emb_reshape, sent_emb_mask,
                                 sent_emb_lens)

        # collect embeddings
        vid_emb_list.extend(vid_emb.detach().cpu())
        par_emb_list.extend(par_emb.detach().cpu())

        #clip-sentence embeddings
        clip_emb_list.extend(clip_emb.detach().cpu())
        sent_emb_list.extend(sent_emb.detach().cpu())

        # logging
        if step % 10 == 0:
            logger.info(f"Val [{step}/{max_step}] Loss {loss.item():.4f}")
    vid_emb_list = torch.stack(vid_emb_list, 0)
    par_emb_list = torch.stack(par_emb_list, 0)
    clip_emb_list = torch.stack(clip_emb_list, 0)
    sent_emb_list = torch.stack(sent_emb_list, 0)
    # video text retrieval
    vid_emb_list = F.normalize(vid_emb_list).numpy()
    par_emb_list = F.normalize(par_emb_list).numpy()

    v2p_res, _ = coot_utils.compute_retr_vid_to_par(vid_emb_list, par_emb_list)
    p2v_res, _ = coot_utils.compute_retr_par_to_vid(vid_emb_list, par_emb_list)
    sum_at_1 = v2p_res["r1"] + p2v_res["r1"]

    logger.info(coot_utils.EVALHEADER)
    logger.info(coot_utils.retrieval_results_to_str(p2v_res, "Par2Vid"))
    logger.info(coot_utils.retrieval_results_to_str(v2p_res, "Vid2Par"))

    # clip sentence retrieval
    clip_emb_list = F.normalize(clip_emb_list).numpy()
    sent_emb_list = F.normalize(sent_emb_list).numpy()

    c2s_res, _ = coot_utils.compute_retr_vid_to_par(clip_emb_list,
                                                    sent_emb_list)
    s2c_res, _ = coot_utils.compute_retr_par_to_vid(clip_emb_list,
                                                    sent_emb_list)
    c2s_sum_at_1 = c2s_res["r1"] + s2c_res["r1"]
    logger.info(coot_utils.EVALHEADER)
    logger.info(coot_utils.retrieval_results_to_str(s2c_res, "Sen2Shot"))
    logger.info(coot_utils.retrieval_results_to_str(c2s_res, "Shot2Sen"))

    writer.add_scalar('val_loss_epoch', loss, epoch)
    writer.add_scalar('val_R1_Sentence2Clip_epoch', s2c_res["r1"], epoch)
    writer.add_scalar('val_R5_Sentence2Clip_acc_epoch', s2c_res["r5"], epoch)
    writer.add_scalar('val_R10_Sentence2Clip_acc_epoch', s2c_res["r10"], epoch)

    writer.add_scalar('val_loss_epoch', loss, epoch)
    writer.add_scalar('val_R1_Clip2Sentence_epoch', c2s_res["r1"], epoch)
    writer.add_scalar('val_R5_Clip2Sentence_acc_epoch', c2s_res["r5"], epoch)
    writer.add_scalar('val_R10_Clip2Sentence_acc_epoch', c2s_res["r10"], epoch)

    return ((v2p_res, p2v_res, sum_at_1), (c2s_res, s2c_res, c2s_sum_at_1))
