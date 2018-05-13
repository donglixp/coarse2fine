import torch
from torch.autograd import Variable

import table
import table.IO
import table.ModelConstructor
import table.Models
import table.modules
from table.Utils import add_pad, argmax
from table.ParseResult import ParseResult

def v_eval(a):
    return Variable(a, volatile=True)


def cpu_vector(v):
    return v.clone().view(-1).cpu()


class Translator(object):
    def __init__(self, opt, dummy_opt):
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

    def translate(self, batch):
        q, q_len = batch.src
        tbl, tbl_len = batch.tbl
        ent, tbl_split, tbl_mask = batch.ent, batch.tbl_split, batch.tbl_mask

        # encoding
        q_enc, q_all, tbl_enc, q_ht, batch_size = self.model.enc(
            q, q_len, ent, tbl, tbl_len, tbl_split, tbl_mask)

        # (1) decoding
        agg_pred = cpu_vector(argmax(self.model.agg_classifier(q_ht).data))
        sel_pred = cpu_vector(argmax(self.model.sel_match(
            q_ht, tbl_enc, tbl_mask).data))
        lay_pred = argmax(self.model.lay_classifier(q_ht).data)
        # get layout op tokens
        op_batch_list = []
        op_idx_batch_list = []
        if self.opt.gold_layout:
            lay_pred = batch.lay.data
            cond_op, cond_op_len = batch.cond_op
            cond_op_len_list = cond_op_len.view(-1).tolist()
            for i, len_it in enumerate(cond_op_len_list):
                if len_it == 0:
                    op_idx_batch_list.append([])
                    op_batch_list.append([])
                else:
                    idx_list = cond_op.data[0:len_it, i].contiguous().view(-1).tolist()
                    op_idx_batch_list.append([int(self.fields['cond_op'].vocab.itos[it]) for it in idx_list])
                    op_batch_list.append(idx_list)
        else:
            lay_batch_list = lay_pred.view(-1).tolist()
            for lay_it in lay_batch_list:
                tk_list = self.fields['lay'].vocab.itos[lay_it].split(' ')
                if (len(tk_list) == 0) or (tk_list[0] == ''):
                    op_idx_batch_list.append([])
                    op_batch_list.append([])
                else:
                    op_idx_batch_list.append([int(op_str) for op_str in tk_list])
                    op_batch_list.append(
                        [self.fields['cond_op'].vocab.stoi[op_str] for op_str in tk_list])
            # -> (num_cond, batch)
            cond_op = v_eval(add_pad(
                op_batch_list, self.fields['cond_op'].vocab.stoi[table.IO.PAD_WORD]).t())
            cond_op_len = torch.LongTensor([len(it) for it in op_batch_list])
        # emb_op -> (num_cond, batch, emb_size)
        if self.model.opt.layout_encode == 'rnn':
            emb_op = table.Models.encode_unsorted_batch(
                self.model.lay_encoder, cond_op, cond_op_len.clamp(min=1))
        else:
            emb_op = self.model.cond_embedding(cond_op)

        # (2) decoding
        self.model.cond_decoder.attn.applyMaskBySeqBatch(q)
        cond_state = self.model.cond_decoder.init_decoder_state(q_all, q_enc)
        cond_col_list, cond_span_l_list, cond_span_r_list = [], [], []
        for emb_op_t in emb_op:
            emb_op_t = emb_op_t.unsqueeze(0)
            cond_context, cond_state, _ = self.model.cond_decoder(
                emb_op_t, q_all, cond_state)
            # cond col -> (1, batch)
            cond_col = argmax(self.model.cond_col_match(
                cond_context, tbl_enc, tbl_mask).data)
            cond_col_list.append(cpu_vector(cond_col))
            # emb_col
            batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(0).cuda().expand(
                cond_col.size(0), cond_col.size(1))
            emb_col = tbl_enc[cond_col, batch_index, :]
            cond_context, cond_state, _ = self.model.cond_decoder(
                emb_col, q_all, cond_state)
            # cond span
            q_mask = v_eval(
                q.data.eq(self.model.pad_word_index).transpose(0, 1))
            cond_span_l = argmax(self.model.cond_span_l_match(
                cond_context, q_all, q_mask).data)
            cond_span_l_list.append(cpu_vector(cond_span_l))
            # emb_span_l: (1, batch, hidden_size)
            emb_span_l = q_all[cond_span_l, batch_index, :]
            cond_span_r = argmax(self.model.cond_span_r_match(
                cond_context, q_all, q_mask, emb_span_l).data)
            cond_span_r_list.append(cpu_vector(cond_span_r))
            # emb_span_r: (1, batch, hidden_size)
            emb_span_r = q_all[cond_span_r, batch_index, :]
            emb_span = self.model.span_merge(
                torch.cat([emb_span_l, emb_span_r], 2))
            cond_context, cond_state, _ = self.model.cond_decoder(
                emb_span, q_all, cond_state)

        # (3) recover output
        indices = cpu_vector(batch.indices.data)
        r_list = []
        for b in range(batch_size):
            idx = indices[b]
            agg = agg_pred[b]
            sel = sel_pred[b]
            cond = []
            for i in range(len(op_batch_list[b])):
                col = cond_col_list[i][b]
                op = op_idx_batch_list[b][i]
                span_l = cond_span_l_list[i][b]
                span_r = cond_span_r_list[i][b]
                cond.append((col, op, (span_l, span_r)))
            r_list.append(ParseResult(idx, agg, sel, cond))

        return r_list
