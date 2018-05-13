"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
from itertools import count
import torch
import torch.nn as nn
from torch.autograd import Variable
import random as rnd

import table
from table.modules.cross_entropy_smooth import CrossEntropyLossSmooth

class LossCompute(nn.Module):
    def __init__(self, smooth_eps=0):
        super(LossCompute, self).__init__()
        self.criterion = {}
        if smooth_eps > 0:
            self.criterion['lay'] = CrossEntropyLossSmooth(
                size_average=False, ignore_index=table.IO.PAD, smooth_eps=smooth_eps)
            self.criterion['tgt'] = CrossEntropyLossSmooth(
                size_average=False, ignore_index=table.IO.PAD, smooth_eps=smooth_eps)
        else:
            self.criterion['lay'] = nn.NLLLoss(
                size_average=False, ignore_index=table.IO.PAD)
            # self.criterion['tgt'] = nn.CrossEntropyLoss(
            self.criterion['tgt'] = nn.NLLLoss(
                size_average=False, ignore_index=table.IO.PAD)
        self.criterion['token'] = nn.BCEWithLogitsLoss(size_average=False)
        self.criterion['cover'] = nn.KLDivLoss(size_average=False)

    def compute_loss(self, pred, gold, mask):
        loss_list = []
        for loss_name in ('lay', 'tgt', 'token'):
            if loss_name not in gold:
                continue
            for i, p, g in zip(count(), pred[loss_name], gold[loss_name]):
                if (loss_name in mask) and mask[loss_name][i] == 1:
                    continue
                loss = self.criterion[loss_name](p, g)
                loss_list.append(loss)

        if 'cover' in gold:
            loss_list.append(gold['cover'])

        # sum up the loss functions
        return sum(loss_list)
