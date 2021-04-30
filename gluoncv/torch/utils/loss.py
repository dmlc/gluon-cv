"""
Loss functions used for training COOT model
"""

from torch import nn
import torch

__all__ = ['MaxMarginRankingLoss', 'CycleConsistencyCootLoss', 'IOULoss']


class MaxMarginRankingLoss(nn.Module):
    """MaxMarginRanking loss used in COOT paper (Eq. 1): https://arxiv.org/abs/2011.00597
    Inputs:
        - x_embedding: the embedding tensor for a modality with shape (Batch x feature_dimension).
        - y_embedding: the embedding tensor for another input modality with same shape of x_embedding.
          i.e. in COOT paper x_embedding could be video features and y_embedding language features.
        - margin: constant margin in eq. 1 of COOT paper.
    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """
    def __init__(self, use_cuda: bool, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.sim = self._cosine_sim
        self.use_cuda = use_cuda

    def _cosine_sim(self, embedding_x1, embedding_x2):
        """ calculate cosine similarity"""
        return embedding_x1.mm(embedding_x2.t())

    def forward(self, x_embedding, y_embedding):
        """ calculate marxmargin ranking loss"""

        scores = self.sim(x_embedding,
                          y_embedding)  # calculate the cosine similarity
        diagonal = scores.diag().view(
            x_embedding.size(0),
            1)  # extract the diagonal scores ( i.e. video_i, sentence_i)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_x1 = (self.margin + scores - d1).clamp(
            min=0)  # First term in eq.1 of COOT paper
        cost_x2 = (self.margin + scores - d2).clamp(
            min=0)  # Second term in eq.1 of COOT paper
        mask = torch.eye(scores.size(0)) > .5

        if self.use_cuda:
            mask = mask.cuda()

        cost_x1 = cost_x1.masked_fill_(mask, 0)
        cost_x2 = cost_x2.masked_fill_(mask, 0)
        return (cost_x1.sum() + cost_x2.sum()).div(x_embedding.shape[0] *
                                                   y_embedding.shape[0])


class CycleConsistencyCootLoss(nn.Module):
    """Compute CycleConsistency loss in COOT paper (Eq. 8)

    Args:
        clip_emb: Embeddings of clips
        clip_mask: Mask
        clip_lens: Number of clips per each video
        sentence_emb: Embeddings of sentences
        sentence_mask: Mask for input language
        sentence_lens: Number of sentences per paragraph

    Outputs:
        - ** clip to clip loss**: loss tensor with shape (batch_size,).
        - ** sentence to sentence loss**: loss tensor with shape (batch_size,).

    """
    def __init__(self, num_samples=-1, use_cuda=True):
        super().__init__()
        self.num_samples = num_samples
        self.use_cuda = use_cuda
        self.num_samples_tensor = (torch.ones(1) * self.num_samples)
        if self.use_cuda:
            self.num_samples_tensor = self.num_samples_tensor.cuda(
                non_blocking=True)
        self.loss_distance_fn = self._compute_mean_distance_l2
        self.proximity_fn = self._compute_mean_distance_negative_l2
        self.proximity_mask_val = -1e18
        self.softmax = nn.Softmax(dim=-1)

    def _compute_mean_distance_l2(self, c, s):
        return torch.mean((c - s)**2, dim=-1)

    def _compute_mean_distance_negative_l2(self, c, s):
        return -self._compute_mean_distance_l2(c, s)

    def forward(self, clip_emb, clip_mask, clip_lens, sentence_emb,
                sentence_mask, sentence_lens):
        """ calculate the loss"""

        clip_max_len, _ = torch.max(clip_lens, dim=-1)
        sentence_max_len, _ = torch.max(sentence_lens, dim=-1)
        clip_sentence_nn, _, _ = self._get_soft_nn(clip_emb, clip_mask,
                                                   sentence_emb, sentence_mask)
        _, clip_beta, _ = self._get_soft_nn(clip_sentence_nn, clip_mask,
                                            clip_emb, clip_mask)
        clip_clip_loss = self._get_loss(clip_mask, clip_lens, clip_max_len,
                                        clip_beta)
        sentence_clip_nn, _, _ = self._get_soft_nn(sentence_emb, sentence_mask,
                                                   clip_emb, clip_mask)
        _, sentence_beta, _ = self._get_soft_nn(sentence_clip_nn,
                                                sentence_mask, sentence_emb,
                                                sentence_mask)
        sentence_sentence_loss = self._get_loss(sentence_mask, sentence_lens,
                                                sentence_max_len,
                                                sentence_beta)
        return clip_clip_loss, sentence_sentence_loss

    def _get_mxn_repr(self, source_emb, source_mask, target_emb, target_mask):
        source_rep = source_emb.unsqueeze(2)
        target_rep = target_emb.unsqueeze(1)
        total_mask = source_mask.unsqueeze(2).bool() & target_mask.unsqueeze(
            1).bool()
        return source_rep, target_rep, total_mask

    def _get_soft_nn(self, source_emb, source_mask, target_emb, target_mask):
        source_rep, target_rep, total_mask = self._get_mxn_repr(
            source_emb, source_mask, target_emb, target_mask)
        distance = self.proximity_fn(source_rep, target_rep)
        distance.masked_fill_(total_mask == 0, self.proximity_mask_val)
        weights_alpha = self.softmax(distance)
        soft_nn = target_emb.unsqueeze(dim=1) * weights_alpha.unsqueeze(dim=3)
        soft_nn = torch.sum(soft_nn, dim=2)
        return soft_nn, weights_alpha, distance

    def _get_loss(self, emb_mask, emb_lens, emb_max_len, beta):
        idx_orig = torch.arange(emb_max_len)
        batch_size, _ = emb_mask.shape
        if self.use_cuda:
            idx_orig = idx_orig.cuda(non_blocking=True)
        idx_orig.unsqueeze_(0)
        index_nn = torch.sum(idx_orig.unsqueeze(1) * beta, dim=-1)
        idx_nn_rep, idx_orig_rep, emb_mask_rep = self._get_mxn_repr(
            index_nn, emb_mask, idx_orig, emb_mask)
        distance = self.loss_distance_fn(idx_nn_rep.unsqueeze(-1),
                                         idx_orig_rep.unsqueeze(-1))
        distance.masked_fill_(emb_mask_rep == 0, 0)
        l_seq = distance.diagonal(dim1=-2, dim2=-1)
        if self.num_samples != -1:
            n_samp = torch.min(emb_lens, self.num_samples_tensor)
            total_loss = 0
            for _, (c_loss, c_mask,
                    c_nsamp) in enumerate(zip(l_seq, emb_mask, n_samp)):
                idx = torch.multinomial(c_mask, int(c_nsamp))
                total_loss += c_loss[idx].mean()
            return total_loss / batch_size

        else:
            return (l_seq.sum(dim=-1) / emb_lens).mean(dim=-1)


class IOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()
