import torch
from torch import nn
import numpy as np


'''
Paper: `Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting`
2, 5, 10, 30 is the thresholds of the precipitation intensity
'''
class BMAEloss(nn.Module):
    def __init__(self):
        super(BMAEloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()
        flag_2 = (a < m).int()
        flag_3 = flag_1 + flag_2
        return flag_3 == 2

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()
        mask[y < 2] = 1
        mask[self.fundFlag(y, 2, 5)] = 2
        mask[self.fundFlag(y, 5, 10)] = 5
        mask[self.fundFlag(y, 10, 30)] = 10
        mask[y > 30] = 30
        return torch.sum(mask * torch.abs(y - pred))


def fundFlag(a, n, m):
    flag_1 = np.uint8(a >= n)
    flag_2 = np.uint8(a < m)
    flag_3 = flag_1 + flag_2
    return flag_3 == 2

def B_mse(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mse = np.sum(mask * ((a - b) ** 2)) / n
    return mse

def B_mae(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mae = np.sum(mask * np.abs(a - b)) / n
    return mae