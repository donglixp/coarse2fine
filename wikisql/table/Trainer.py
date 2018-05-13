"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
"""
from __future__ import division
import os
import time
import sys
import math
import torch
import torch.nn as nn

import table
import table.modules
from table.Utils import argmax


class Statistics(object):
    def __init__(self, loss, eval_result):
        self.loss = loss
        self.eval_result = eval_result
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        for k, v in stat.eval_result.items():
            if k in self.eval_result:
                v0 = self.eval_result[k][0] + v[0]
                v1 = self.eval_result[k][1] + v[1]
                self.eval_result[k] = (v0, v1)
            else:
                self.eval_result[k] = (v[0], v[1])

    def accuracy(self, return_str=False):
        d = sorted([(k, v)
                    for k, v in self.eval_result.items()], key=lambda x: x[0])
        if return_str:
            return '; '.join((('{}: {:.2%}'.format(k, v[0] / v[1],)) for k, v in d))
        else:
            return dict([(k, 100.0 * v[0] / v[1]) for k, v in d])

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        print(("Epoch %2d, %5d/%5d; %s; %.0f s elapsed") %
              (epoch, batch, n_batches, self.accuracy(True), time.time() - start))
        sys.stdout.flush()

    def log(self, split, logger, lr, step):
        pass


def count_accuracy(scores, target, mask=None, row=False):
    pred = argmax(scores)
    if mask is None:
        m_correct = pred.eq(target)
        num_all = m_correct.numel()
    elif row:
        m_correct = pred.eq(target).masked_fill_(
            mask, 1).prod(0, keepdim=False)
        num_all = m_correct.numel()
    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum()
    return (m_correct, num_all)


def aggregate_accuracy(r_dict, metric_name_list):
    m_list = []
    for metric_name in metric_name_list:
        m_list.append(r_dict[metric_name][0])
    agg = torch.stack(m_list, 0).prod(0, keepdim=False)
    return (agg.sum(), agg.numel())


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim

        # Set model in training mode.
        self.model.train()

    def forward(self, batch, criterion):
        # 1. F-prop.
        q, q_len = batch.src
        tbl, tbl_len = batch.tbl
        cond_op, cond_op_len = batch.cond_op
        agg_out, sel_out, lay_out, cond_col_out, cond_span_l_out, cond_span_r_out = self.model(
            q, q_len, batch.ent, tbl, tbl_len, batch.tbl_split, batch.tbl_mask, cond_op, cond_op_len, batch.cond_col, batch.cond_span_l, batch.cond_span_r, batch.lay)

        # 2. Compute loss.
        pred = {'agg': agg_out, 'sel': sel_out, 'lay': lay_out, 'cond_col': cond_col_out,
                'cond_span_l': cond_span_l_out, 'cond_span_r': cond_span_r_out}
        gold = {'agg': batch.agg, 'sel': batch.sel, 'lay': batch.lay, 'cond_col': batch.cond_col_loss,
                'cond_span_l': batch.cond_span_l_loss, 'cond_span_r': batch.cond_span_r_loss}
        loss = criterion.compute_loss(pred, gold)

        # 3. Get the batch statistics.
        r_dict = {}
        for metric_name in ('agg', 'sel', 'lay'):
            r_dict[metric_name] = count_accuracy(
                pred[metric_name].data, gold[metric_name].data)
        for metric_name in ('cond_col', 'cond_span_l', 'cond_span_r'):
            r_dict[metric_name + '-token'] = count_accuracy(
                pred[metric_name].data, gold[metric_name].data, mask=gold[metric_name].data.eq(-1), row=False)
            r_dict[metric_name] = count_accuracy(
                pred[metric_name].data, gold[metric_name].data, mask=gold[metric_name].data.eq(-1), row=True)
        st = dict([(k, (v[0].sum(), v[1])) for k, v in r_dict.items()])
        st['where'] = aggregate_accuracy(
            r_dict, ('lay', 'cond_col', 'cond_span_l', 'cond_span_r'))
        st['all'] = aggregate_accuracy(
            r_dict, ('agg', 'sel', 'lay', 'cond_col', 'cond_span_l', 'cond_span_r'))
        batch_stats = Statistics(loss.data[0], st)

        return loss, batch_stats

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics(0, {})
        report_stats = Statistics(0, {})

        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()

            loss, batch_stats = self.forward(batch, self.train_loss)

            # Update the parameters and statistics.
            loss.backward()
            self.optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if report_func is not None:
                report_stats = report_func(
                    epoch, i, len(self.train_iter),
                    total_stats.start_time, self.optim.lr, report_stats)

        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics(0, {})
        for batch in self.valid_iter:
            loss, batch_stats = self.forward(batch, self.valid_loss)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, eval_metric, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(eval_metric, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """

        model_state_dict = self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'model': model_state_dict,
            'vocab': table.IO.TableDataset.save_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim
        }
        eval_result = valid_stats.accuracy()
        torch.save(checkpoint, os.path.join(
            opt.save_path, 'm_%d.pt' % (epoch)))
