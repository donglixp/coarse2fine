"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import random as rnd

import table
from table.modules.cross_entropy_smooth import CrossEntropyLossSmooth


class TableLossCompute(nn.Module):
    def __init__(self, agg_sample_rate, smooth_eps=0):
        super(TableLossCompute, self).__init__()
        self.criterion = {}
        nll = nn.NLLLoss(size_average=False, ignore_index=-1)
        if smooth_eps > 0:
            for loss_name in ('sel', 'cond_col', 'cond_span_l', 'cond_span_r'):
                self.criterion[loss_name] = nll
            for loss_name in ('agg', 'lay'):
                self.criterion[loss_name] = CrossEntropyLossSmooth(
                    size_average=False, ignore_index=-1, smooth_eps=smooth_eps)
        else:
            for loss_name in ('agg', 'sel', 'lay', 'cond_col', 'cond_span_l', 'cond_span_r'):
                self.criterion[loss_name] = nll
        self.agg_sample_rate = agg_sample_rate

    def compute_loss(self, pred, gold):
        # sum up the loss functions
        loss_list = []
        for loss_name in ('agg', 'sel', 'lay'):
            loss = self.criterion[loss_name](pred[loss_name], gold[loss_name])
            if (loss_name != 'agg') or (rnd.random() < self.agg_sample_rate):
                loss_list.append(loss)
        for loss_name in ('cond_col', 'cond_span_l', 'cond_span_r'):
            for p, g in zip(pred[loss_name], gold[loss_name]):
                loss = self.criterion[loss_name](p, g)
                loss_list.append(loss)
        return sum(loss_list)
