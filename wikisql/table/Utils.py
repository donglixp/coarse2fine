import os
from path import Path
import torch
import random
import numpy as np
from collections import defaultdict


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def sort_for_pack(input_len):
    idx_sorted, input_len_sorted = zip(
        *sorted(list(enumerate(input_len)), key=lambda x: x[1], reverse=True))
    idx_sorted, input_len_sorted = list(idx_sorted), list(input_len_sorted)
    idx_map_back = list(map(lambda x: x[0], sorted(
        list(enumerate(idx_sorted)), key=lambda x: x[1])))
    return idx_sorted, input_len_sorted, idx_map_back


def argmax(scores):
    return scores.max(scores.dim() - 1)[1]


def add_pad(b_list, pad_index, return_tensor=True):
    max_len = max((len(b) for b in b_list))
    r_list = []
    for b in b_list:
        r_list.append(b + [pad_index] * (max_len - len(b)))
    if return_tensor:
        return torch.LongTensor(r_list).cuda()
    else:
        return r_list
