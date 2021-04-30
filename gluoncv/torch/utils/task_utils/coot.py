# pylint: disable=line-too-long
"""
Utility functions for task
"""
from timeit import default_timer as timer

import torch
from torch.nn import functional as F

from .. import coot_utils
from ..coot_utils import compute_constrastive_loss, compute_cmc_loss
from ..coot_utils import unpack_data

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
