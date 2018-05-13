import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

import table
import table.IO
import table.ModelConstructor
import table.Models
import table.modules
from table.Utils import add_pad, argmax
from table.ParseResult import ParseResult
from table.Models import encode_unsorted_batch
from tree import SKIP_OP_LIST
from bpe import recover_bpe


def v_eval(a):
    return Variable(a, volatile=True)


def cpu_vector(v):
    return v.clone().view(-1).cpu()


def recover_layout_token(pred_list, vocab, max_sent_length):
    r_list = []
    for i in range(max_sent_length):
        r_list.append(vocab.itos[pred_list[i]])
        if r_list[-1] == table.IO.EOS_WORD:
            r_list = r_list[:-1]
            break
    return r_list


def mix_lay_and_tgt(lay_skip, tgt):
    if len(lay_skip) == len(tgt):
        tgt_mix = []
        for tk_lay, tk_tgt in zip(lay_skip, tgt):
            if tk_lay in (table.IO.TOK_WORD, table.IO.SKP_WORD):
                tgt_mix.append(tk_tgt)
            else:
                tgt_mix.append(tk_lay)
        return tgt_mix
    else:
        return tgt


def recover_target_token(lay_skip, pred_list, vocab, max_sent_length):
    r_list = []
    for i in range(min(len(lay_skip), len(pred_list))):
        if lay_skip[i] in (table.IO.TOK_WORD, table.IO.SKP_WORD):
            r_list.append(vocab.itos[pred_list[i]])
        elif lay_skip[i] in (table.IO.RIG_WORD,):
            r_list.append(')')
        else:
            r_list.append(lay_skip[i])
    return r_list


def get_decode_batch_length(dec, batch_size, max_sent_length):
    r_list = []
    for b in range(batch_size):
        find_len = None
        for i in range(max_sent_length):
            if dec[i, b] == table.IO.EOS:
                find_len = i
                break
        if find_len is None:
            r_list.append(max_sent_length)
        else:
            r_list.append(find_len)
    assert(len(r_list) == batch_size)
    return torch.LongTensor(r_list)


# ['(airline:e@1', '(argmin', '(and', 'flight@1', 'from@2', 'to@2', 'day_number@2', 'month@2', ')', 'fare@1', ')', ')']
def expand_layout_with_skip(lay_list):
    lay_skip_list, tgt_mask_list, lay_index_list = [], [], []
    for lay in lay_list:
        lay_skip = []
        for tk_lay in lay:
            if len(tk_lay) >= 2 and tk_lay[-2] == '@':
                op = tk_lay[:-2]
                if tk_lay.startswith('('):
                    lay_skip.append(op)
                else:
                    # need to expand
                    k = int(tk_lay[-1])
                    # ')' can be generated according to layout rather than predicting
                    lay_skip.extend(
                        ['(' + op] + [table.IO.SKP_WORD for __ in range(k)] + [table.IO.RIG_WORD])
            else:
                lay_skip.append(tk_lay)
                if tk_lay[1:] in SKIP_OP_LIST:
                    lay_skip.append(table.IO.SKP_WORD)
        lay_skip_list.append(lay_skip)
        # tgt_mask
        tgt_mask_list.append(table.IO.get_tgt_mask(lay_skip))
        # lay_index
        lay_index_list.append(table.IO.get_lay_index(lay_skip))
    tgt_mask_seq = add_pad(tgt_mask_list, 1).float().t()
    lay_index_seq = add_pad(lay_index_list, 0).t()
    return lay_skip_list, tgt_mask_seq, lay_index_seq


class Translator(object):
    def __init__(self, opt, dummy_opt={}):
        # Add in default model arguments, possibly added since training.
        self.opt = opt
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)
        self.fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

        model_opt = checkpoint['opt']
        model_opt.pre_word_vecs = opt.pre_word_vecs
        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]

        self.model = table.ModelConstructor.make_base_model(
            model_opt, self.fields, checkpoint)
        self.model.eval()

        if model_opt.moving_avg > 0:
            for p, avg_p in zip(self.model.parameters(), checkpoint['moving_avg']):
                p.data.copy_(avg_p)

        if opt.attn_ignore_small > 0:
            self.model.lay_decoder.attn.ignore_small = opt.attn_ignore_small
            self.model.tgt_decoder.attn.ignore_small = opt.attn_ignore_small

    def _init_parent_list(self, decoder, q_enc, batch_size):
        # (num_layers * num_directions, batch, hidden_size)
        q_ht, q_ct = q_enc
        q_ht = q_ht[-1] if not self.model.opt.brnn else q_ht[-2:].transpose(
            0, 1).contiguous().view(batch_size, -1)
        if self.model.opt.parent_feed == 'output':
            parent_list = [[q_ht[b].unsqueeze(0)] for b in range(batch_size)]
        elif self.model.opt.parent_feed == 'input':
            decoder.init_parent_all(q_ht.unsqueeze(0))
            parent_list = [[0] for b in range(batch_size)]
        else:
            parent_list = None
        return parent_list

    def _cat_parent_feed_input(self, parent_list, batch_size):
        parent_index = v_eval(torch.LongTensor(
            [parent_list[b][-1] for b in range(batch_size)]).unsqueeze_(0).cuda())
        return parent_index

    def _cat_parent_feed_output(self, dec_all, parent_list, batch_size):
        parent_feed = torch.stack(
            [parent_list[b][-1] for b in range(batch_size)], 1)
        # -> (dec_seq_len, batch_size, 2 * rnn_size)
        dec_all = torch.cat([dec_all, parent_feed], 2)
        return dec_all

    def _update_parent_list(self, i, parent_list, dec_rnn_output, inp_cpu, lay_skip_list, vocab, batch_size):
        # append to parent_list
        for b in range(batch_size):
            tk = vocab.itos[inp_cpu[b]]
            if (lay_skip_list is not None) and (i < len(lay_skip_list[b])):
                lay_skip = lay_skip_list[b]
                if lay_skip[i] in (table.IO.TOK_WORD, table.IO.SKP_WORD):
                    pass
                elif lay_skip[i] in (table.IO.RIG_WORD,):
                    tk = ')'
                else:
                    tk = lay_skip[i]
            if tk.startswith('('):
                if self.model.opt.parent_feed == 'output':
                    parent_list[b].append(dec_rnn_output[:, b, :])
                elif self.model.opt.parent_feed == 'input':
                    parent_list[b].append(i + 1)
            elif tk == ')':
                if len(parent_list[b]) > 1:
                    parent_list[b].pop()

    def run_lay_decoder(self, decoder, classifier, q, q_all, q_enc, max_dec_len, vocab_mask, vocab):
        batch_size = q.size(1)
        decoder.attn.applyMaskBySeqBatch(q)
        dec_list = []
        dec_state = decoder.init_decoder_state(q_all, q_enc)
        inp = torch.LongTensor(1, batch_size).fill_(table.IO.BOS).cuda()
        if self.model.opt.parent_feed in ('input', 'output'):
            parent_list = self._init_parent_list(decoder, q_enc, batch_size)
        for i in range(max_dec_len):
            inp = v_eval(inp)
            if self.model.opt.parent_feed == 'input':
                parent_index = self._cat_parent_feed_input(
                    parent_list, batch_size)
            else:
                parent_index = None
            dec_all, dec_state, _, dec_rnn_output = decoder(
                inp, q_all, dec_state, parent_index)
            if self.model.opt.parent_feed == 'output':
                dec_all = self._cat_parent_feed_output(
                    dec_all, parent_list, batch_size)
            dec_all = dec_all.view(batch_size, -1)
            dec_out = classifier(dec_all)
            dec_out = dec_out.data.view(1, batch_size, -1)
            if vocab_mask is not None:
                dec_out_part = dec_out[:, :, len(table.IO.special_token_list):]
                dec_out_part.masked_fill_(vocab_mask, -float('inf'))
                # dec_out_part.masked_scatter_(vocab_mask, dec_out_part[vocab_mask].add(-math.log(1000)))
            inp = argmax(dec_out)
            # topk = [vocab.itos[idx] for idx in dec_out[0, 0, :].topk(10, dim=0)[1]]
            # print(topk)
            inp_cpu = cpu_vector(inp)
            dec_list.append(inp_cpu)
            if self.model.opt.parent_feed in ('input', 'output'):
                self._update_parent_list(
                    i, parent_list, dec_rnn_output, inp_cpu, None, vocab, batch_size)
        return torch.stack(dec_list, 0)

    def run_tgt_decoder(self, embeddings, tgt_mask_seq, lay_index_seq, lay_all, decoder, classifier, q, q_all, q_enc, max_dec_len, lay_skip_list, vocab):
        batch_size = q.size(1)
        decoder.attn.applyMaskBySeqBatch(q)
        dec_list = []
        dec_state = decoder.init_decoder_state(q_all, q_enc)
        inp = torch.LongTensor(1, batch_size).fill_(table.IO.BOS).cuda()
        batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(0).cuda()
        if self.model.opt.parent_feed in ('input', 'output'):
            parent_list = self._init_parent_list(decoder, q_enc, batch_size)
        for i in range(min(max_dec_len, lay_index_seq.size(0))):
            # (1, batch)
            lay_index = lay_index_seq[i].unsqueeze(0)
            lay_select = lay_all[lay_index, batch_index, :]
            tgt_inp_emb = embeddings(v_eval(inp))
            tgt_mask_expand = v_eval(tgt_mask_seq[i].unsqueeze(
                0).unsqueeze(2).expand_as(tgt_inp_emb))
            inp = tgt_inp_emb.mul(tgt_mask_expand) + \
                lay_select.mul(1 - tgt_mask_expand)
            if self.model.opt.parent_feed == 'input':
                parent_index = self._cat_parent_feed_input(
                    parent_list, batch_size)
            else:
                parent_index = None
            dec_all, dec_state, _, dec_rnn_output = decoder(
                inp, q_all, dec_state, parent_index)
            if self.model.opt.parent_feed == 'output':
                dec_all = self._cat_parent_feed_output(
                    dec_all, parent_list, batch_size)
            dec_all = dec_all.view(batch_size, -1)
            dec_out = classifier(dec_all)
            dec_out = dec_out.view(1, batch_size, -1)
            inp = argmax(dec_out.data)
            # RIG_WORD -> ')'
            rig_mask = []
            for b in range(batch_size):
                tk = lay_skip_list[b][i] if i < len(lay_skip_list[b]) else None
                rig_mask.append(1 if tk in (table.IO.RIG_WORD,) else 0)
            inp.masked_fill_(torch.ByteTensor(
                rig_mask).unsqueeze_(0).cuda(), vocab.stoi[')'])
            inp_cpu = cpu_vector(inp)
            dec_list.append(inp_cpu)
            if self.model.opt.parent_feed in ('input', 'output'):
                self._update_parent_list(
                    i, parent_list, dec_rnn_output, inp_cpu, lay_skip_list, vocab, batch_size)
        return torch.stack(dec_list, 0)

    def translate(self, batch):
        q, q_len = batch.src
        batch_size = q.size(1)

        # encoding
        q_enc, q_all = self.model.q_encoder(q, lengths=q_len, ent=None)
        if self.model.opt.seprate_encoder:
            q_tgt_enc, q_tgt_all = self.model.q_tgt_encoder(
                q, lengths=q_len, ent=None)
        else:
            q_tgt_enc, q_tgt_all = q_enc, q_all

        if self.model.opt.layout_token_prune:
            layout_token_prune_list = []
            q_token_enc, __ = self.model.q_token_encoder(
                q, lengths=q_len, ent=None)
            # (num_layers * num_directions, batch, hidden_size)
            q_token_ht, __ = q_token_enc
            batch_size = q_token_ht.size(1)
            q_token_ht = q_token_ht[-1] if not self.model.opt.brnn else q_token_ht[-2:].transpose(
                0, 1).contiguous().view(batch_size, -1)
            # without .t()
            token_out = F.sigmoid(self.model.token_pruner(q_token_ht))
            # decide prune which tokens
            vocab_mask = token_out.data.lt(0).view(1, batch_size, -1)
            for tk_idx in range(len(table.IO.special_token_list), len(self.fields['lay'].vocab)):
                w = self.fields['lay'].vocab.itos[tk_idx]
                if w.startswith('(') or w in (')', table.IO.TOK_WORD):
                    idx = tk_idx - len(table.IO.special_token_list)
                    vocab_mask[:, :, idx] = 0
            # log pruned tokens for evaluation
            for b in range(batch_size):
                masked_v_list = []
                for i in range(vocab_mask.size(2)):
                    if vocab_mask[0, b, i] == 1:
                        masked_v_list.append(
                            self.fields['lay'].vocab.itos[i + len(table.IO.special_token_list)])
                layout_token_prune_list.append(masked_v_list)
        else:
            token_out = None
            vocab_mask = None
            layout_token_prune_list = [None for b in range(batch_size)]

        # layout decoding
        lay_dec = self.run_lay_decoder(
            self.model.lay_decoder, self.model.lay_classifier, q, q_all, q_enc, self.opt.max_lay_len, vocab_mask, self.fields['lay'].vocab)
        if self.opt.gold_layout:
            if self.model.opt.bpe:
                lay_dec = batch.lay_bpe[0].data[1:]
            else:
                lay_dec = batch.lay[0].data[1:]
        # recover layout
        lay_list = []
        for b in range(batch_size):
            if self.model.opt.bpe:
                lay_field = 'lay_bpe'
            else:
                lay_field = 'lay'
            lay = recover_layout_token([lay_dec[i, b] for i in range(
                lay_dec.size(0))], self.fields[lay_field].vocab, lay_dec.size(0))
            if self.model.opt.bpe:
                lay = recover_bpe(lay)
            lay_list.append(lay)

        # layout encoding
        # lay_len = get_decode_batch_length(lay_dec, batch_size, self.opt.max_lay_len)
        lay_len = torch.LongTensor([len(lay_list[b]) for b in range(batch_size)])
        # data used for layout encoding
        lay_dec = torch.LongTensor(
            lay_len.max(), batch_size).fill_(table.IO.PAD)
        for b in range(batch_size):
            for i in range(lay_len[b]):
                lay_dec[i, b] = self.fields['lay'].vocab.stoi[lay_list[b][i]]
        lay_dec = v_eval(lay_dec.cuda())
        # (lay_len, batch, lay_size)
        if self.model.opt.no_lay_encoder:
            lay_all = self.model.lay_encoder(lay_dec)
        else:
            lay_enc_len = lay_len.cuda().clamp(min=1)
            lay_all = encode_unsorted_batch(
                self.model.lay_encoder, lay_dec, lay_enc_len)
        # co-attention
        if self.model.lay_co_attention is not None:
            lay_all = self.model.lay_co_attention(
                lay_all, lay_enc_len, q_all, q)

        # get lay_index and tgt_mask: (tgt_len, batch)
        lay_skip_list, tgt_mask_seq, lay_index_seq = expand_layout_with_skip(
            lay_list)

        # co-attention
        if self.model.q_co_attention is not None:
            q_tgt_enc, q_tgt_all = self.model.q_co_attention(
                q_tgt_all, q_len, lay_all, lay_dec)

        # target decoding
        tgt_dec = self.run_tgt_decoder(self.model.tgt_embeddings, tgt_mask_seq, lay_index_seq, lay_all, self.model.tgt_decoder,
                                       self.model.tgt_classifier, q, q_tgt_all, q_tgt_enc, self.opt.max_tgt_len, lay_skip_list, self.fields['tgt'].vocab)
        # recover target
        tgt_list = []
        for b in range(batch_size):
            tgt = recover_target_token(lay_skip_list[b], [tgt_dec[i, b] for i in range(
                tgt_dec.size(0))], self.fields['tgt'].vocab, tgt_dec.size(0))
            tgt_list.append(tgt)

        # (3) recover output
        indices = cpu_vector(batch.indices.data)
        return [ParseResult(idx, lay, tgt, token_prune)
                for idx, lay, tgt, token_prune in zip(indices, lay_list, tgt_list, layout_token_prune_list)]
