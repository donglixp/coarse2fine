# -*- coding: utf-8 -*-

import os
import argparse
import codecs
import torch

import table
import table.IO
import opts
from table.Utils import set_seed

parser = argparse.ArgumentParser(description='preprocess.py')


# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-train_anno', default="train.jsonl",
                    help="Path to the training annotated data")
parser.add_argument('-valid_anno', default="dev.jsonl",
                    help="Path to the validation annotated data")
parser.add_argument('-test_anno', default="test.jsonl",
                    help="Path to the test annotated data")

parser.add_argument('-save_data', default="",
                    help="Output file for the prepared data")

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


def main():
    print('Preparing training ...')
    fields = table.IO.TableDataset.get_fields()
    print("Building Training...")
    train = table.IO.TableDataset(opt.train_anno, fields, opt, True)

    print("Building Valid...")
    valid = table.IO.TableDataset(opt.valid_anno, fields, opt, True)

    print("Building Test...")
    test = table.IO.TableDataset(opt.test_anno, fields, opt, False)

    print("Building Vocab...")
    table.IO.TableDataset.build_vocab(train, valid, test, opt)

    print("Saving train/valid/fields")
    # Can't save fields, so remove/reconstruct at training time.
    torch.save(table.IO.TableDataset.save_vocab(fields),
               open(os.path.join(opt.save_data, 'vocab.pt'), 'wb'))
    train.fields = []
    valid.fields = []
    torch.save(train, open(os.path.join(opt.save_data, 'train.pt'), 'wb'))
    torch.save(valid, open(os.path.join(opt.save_data, 'valid.pt'), 'wb'))


if __name__ == "__main__":
    main()
