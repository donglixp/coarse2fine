# -*- coding: utf-8 -*-

import os
import argparse
import torch
from path import Path

import table
import table.IO
import opts
from table.Utils import set_seed
from bpe import learn_bpe, BpeProcessor
from tree import STree

parser = argparse.ArgumentParser(description='preprocess.py')


# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-seed', type=int, default=123,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opts.preprocess_opts(parser)

opt = parser.parse_args()
set_seed(opt.seed)

opt.train_anno = os.path.join(opt.root_dir, opt.dataset, 'train.json')
opt.valid_anno = os.path.join(opt.root_dir, opt.dataset, 'dev.json')
opt.test_anno = os.path.join(opt.root_dir, opt.dataset, 'test.json')
opt.save_data = os.path.join(opt.root_dir, opt.dataset)


def main():
    js_list = table.IO.read_anno_json(opt.train_anno, opt)
    bpe_list = learn_bpe([STree(js['lay'])
                          for js in js_list], opt.bpe_num_merge, opt.bpe_min_freq)
    bpe_processor = BpeProcessor(bpe_list, True)

    print('Preparing training ...')
    fields = table.IO.TableDataset.get_fields()
    print("Building Training...")
    train = table.IO.TableDataset(
        opt.train_anno, fields, bpe_processor, opt.permute_order, opt, True)

    if Path(opt.valid_anno).exists():
        print("Building Valid...")
        valid = table.IO.TableDataset(
            opt.valid_anno, fields, bpe_processor, 0, opt, True)
    else:
        valid = None

    if Path(opt.test_anno).exists():
        print("Building Test...")
        test = table.IO.TableDataset(
            opt.test_anno, fields, bpe_processor, 0, opt, False)
    else:
        test = None

    print("Building Vocab...")
    table.IO.TableDataset.build_vocab(train, valid, test, opt)

    print("Saving train/valid/fields")
    # Can't save fields, so remove/reconstruct at training time.
    torch.save(table.IO.TableDataset.save_vocab(fields),
               open(os.path.join(opt.save_data, 'vocab.pt'), 'wb'))
    torch.save(bpe_processor, open(
        os.path.join(opt.save_data, 'bpe.pt'), 'wb'))
    train.fields = []
    torch.save(train, open(os.path.join(opt.save_data, 'train.pt'), 'wb'))

    if Path(opt.valid_anno).exists():
        valid.fields = []
        torch.save(valid, open(os.path.join(opt.save_data, 'valid.pt'), 'wb'))

    if Path(opt.test_anno).exists():
        test.fields = []
        torch.save(test, open(os.path.join(opt.save_data, 'test.pt'), 'wb'))


if __name__ == "__main__":
    main()
