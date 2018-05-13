import torch
import torch.nn as nn
from torch.autograd import Variable


class LockedDropout(nn.Module):
    def __init__(self, dropout_rate):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x
