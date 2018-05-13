import torch
import torch.nn as nn
from torch.autograd import Variable


class PartUpdateEmbedding(nn.Module):
    def __init__(self, update_index, emb_update, emb_fixed):
        super(PartUpdateEmbedding, self).__init__()
        self.update_index = update_index
        self.emb_update = emb_update
        self.emb_fixed = emb_fixed
        self.should_update = True
        self.embedding_dim = emb_update.embedding_dim

    def set_update(self, should_update):
        self.should_update = should_update

    def forward(self, inp):
        assert(inp.dim() == 2)
        r_update = self.emb_update(inp.clamp(0, self.update_index - 1))
        r_fixed = self.emb_fixed(inp)
        mask = Variable(inp.data.lt(self.update_index).float().unsqueeze(
            2).expand_as(r_update), requires_grad=False)
        r_update = r_update.mul(mask)
        r_fixed = r_fixed.mul(1 - mask)
        if self.should_update:
            return r_update + r_fixed
        else:
            return r_update + Variable(r_fixed.data, requires_grad=False)
