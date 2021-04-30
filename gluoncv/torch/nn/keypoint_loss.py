"""Loss layers for keypoints that can be inserted to modules"""
import torch
import torch.nn as nn

__all__ = ['WeightedMSELoss', 'HMFocalLoss']

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

class WeightedMSELoss(nn.Module):
    """Weighted MSE loss layer"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt) **2) * mask
        loss = loss.mean()
        return loss

class HMFocalLoss(nn.Module):
    """Heatmap Focal Loss layer"""
    def __init__(self, alpha, beta):
        super(HMFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        pred = _sigmoid(pred)
        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        else:
            return -(pos_loss + neg_loss) / num_pos
