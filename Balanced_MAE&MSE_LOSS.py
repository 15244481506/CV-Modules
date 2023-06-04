import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


'''
Paper: `VPTR: Efficient Transformers for Video Prediction`
'''
def temporal_weight_func(T):
    t = torch.linspace(0, T-1, T)
    beta = np.log(T)/(T-1)
    w = torch.exp(beta * t)

    return w

class L1Loss(nn.Module):
    def __init__(self, temporal_weight = None, norm_dim = None):
        """
        Args:
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.norm_dim = norm_dim
    
    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.abs(pred - gt)
        if self.temporal_weight is not None:
            w = self.temporal_weight.to(se.device)
            if len(se.shape) == 5:
                se = se * w[None, :, None, None, None]
            elif len(se.shape) == 6:
                se = se * w[None, :, None, None, None, None] #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
        mse = se.mean()
        return mse

class MSELoss(nn.Module):
    def __init__(self, temporal_weight = None, norm_dim = None):
        """
        Args:
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.norm_dim = norm_dim
    
    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.square(pred - gt)
        if self.temporal_weight is not None:
            w = self.temporal_weight.to(se.device)
            if len(se.shape) == 5:
                se = se * w[None, :, None, None, None]
            elif len(se.shape) == 6:
                se = se * w[None, :, None, None, None, None] #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
        mse = se.mean()
        return mse