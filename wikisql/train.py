from __future__ import division

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch import cuda

import table
import table.Models
import table.ModelConstructor
import table.modules
from table.Utils import set_seed
import opts
from tensorboard_logger import Logger
from path import Path


def get_save_index(save_dir):
    save_index = 0
    while True:
        if Path(os.path.join(save_dir, 'run.%d' % (save_index,))).exists():
            save_index += 1
        else:
            break
    return save_index


parser = argparse.ArgumentParser(description='train.py')

# opts.py
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()

opt.save_path = os.path.join(opt.save_dir, 'run.%d' %
                             (get_save_index(opt.save_dir),))
Path(opt.save_path).mkdir_p()

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
opt.pre_word_vecs = os.path.join(opt.data, 'embedding')

print(vars(opt))
json.dump(opt.__dict__, open(os.path.join(
    opt.save_path, 'opt.json'), 'w'), sort_keys=True, indent=2)

cuda.set_device(opt.gpuid[0])
set_seed(opt.seed)

# Set up the logging server.
# logger = Logger(os.path.join(opt.save_path, 'tb'))


def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = table.Statistics(0, {})

    return report_stats


def train_model(model, train_data, valid_data, fields, optim):
    train_iter = table.IO.OrderedIterator(
        dataset=train_data, batch_size=opt.batch_size, device=opt.gpuid[0], repeat=False)
    valid_iter = table.IO.OrderedIterator(
        dataset=valid_data, batch_size=opt.batch_size, device=opt.gpuid[0], train=False, sort=True, sort_within_batch=False)

    train_loss = table.Loss.TableLossCompute(opt.agg_sample_rate, smooth_eps=model.opt.smooth_eps).cuda()
    valid_loss = table.Loss.TableLossCompute(opt.agg_sample_rate, smooth_eps=model.opt.smooth_eps).cuda()

    trainer = table.Trainer(model, train_iter, valid_iter,
                            train_loss, valid_loss, optim)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        if opt.fix_word_vecs:
            if (epoch >= opt.update_word_vecs_after):
                model.q_encoder.embeddings.set_update(True)
            else:
                model.q_encoder.embeddings.set_update(False)

        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, report_func)
        print('Train accuracy: %s' % train_stats.accuracy(True))

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation accuracy: %s' % valid_stats.accuracy(True))

        # 3. Log to remote server.
        # train_stats.log("train", logger, optim.lr, epoch)
        # valid_stats.log("valid", logger, optim.lr, epoch)

        # 4. Update the learning rate
        trainer.epoch_step(None, epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(opt, epoch, fields, valid_stats)


def load_fields(train, valid, checkpoint):
    fields = table.IO.TableDataset.load_fields(
        torch.load(os.path.join(opt.data, 'vocab.pt')))
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

    return fields


def build_model(model_opt, fields, checkpoint):
    print('Building model...')
    model = table.ModelConstructor.make_base_model(
        model_opt, fields, checkpoint)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = table.Optim(
            opt.optim, opt.learning_rate, opt.alpha, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )

    optim.set_parameters(model.parameters())

    return optim


def main():
    # Load train and validate data.
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(os.path.join(opt.data, 'train.pt'))
    valid = torch.load(os.path.join(opt.data, 'valid.pt'))
    print(' * number of training sentences: %d' % len(train))
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(
            opt.train_from, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Load fields generated from preprocess phase.
    fields = load_fields(train, valid, checkpoint)

    # Build model.
    model = build_model(model_opt, fields, checkpoint)

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_model(model, train, valid, fields, optim)


if __name__ == "__main__":
    main()
