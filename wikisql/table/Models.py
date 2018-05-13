from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import table
from table.Utils import aeq, sort_for_pack


def _build_rnn(rnn_type, input_size, hidden_size, num_layers, dropout, weight_dropout, bidirectional=False):
    dr = 0 if weight_dropout > 0 else dropout
    rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                num_layers=num_layers, dropout=dr, bidirectional=bidirectional)
    if weight_dropout > 0:
        param_list = ['weight_hh_l0']
        if bidirectional:
            param_list += [it + '_reverse' for it in param_list]
        rnn = table.modules.WeightDrop(rnn, param_list, dropout=weight_dropout)
    return rnn


class RNNEncoder(nn.Module):
    """ The standard RNN encoder. """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, lock_dropout, weight_dropout, embeddings, ent_embedding):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.ent_embedding = ent_embedding
        self.no_pack_padded_seq = False
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout)
        else:
            self.word_dropout = nn.Dropout(dropout)

        # Use pytorch version when available.
        input_size = embeddings.embedding_dim
        if ent_embedding is not None:
            input_size += ent_embedding.embedding_dim
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size // num_directions, num_layers, dropout, weight_dropout, bidirectional)

    def forward(self, input, lengths=None, hidden=None, ent=None):
        emb = self.embeddings(input)
        if self.ent_embedding is not None:
            emb_ent = self.ent_embedding(ent)
            emb = torch.cat((emb, emb_ent), 2)
        if self.word_dropout is not None:
            emb = self.word_dropout(emb)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        need_pack = (lengths is not None) and (not self.no_pack_padded_seq)
        if need_pack:
            # Lengths data is wrapped inside a Variable.
            if not isinstance(lengths, list):
                lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if need_pack:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


def encode_unsorted_batch(encoder, tbl, tbl_len):
    # sort for pack()
    idx_sorted, tbl_len_sorted, idx_map_back = sort_for_pack(tbl_len)
    tbl_sorted = tbl.index_select(1, Variable(
        torch.LongTensor(idx_sorted).cuda(), requires_grad=False))
    # tbl_context: (seq_len, batch, hidden_size * num_directions)
    __, tbl_context = encoder(tbl_sorted, tbl_len_sorted)
    # recover the sort for pack()
    v_idx_map_back = Variable(torch.LongTensor(
        idx_map_back).cuda(), requires_grad=False)
    tbl_context = tbl_context.index_select(1, v_idx_map_back)
    return tbl_context


class TableRNNEncoder(nn.Module):
    def __init__(self, encoder, split_type='incell', merge_type='cat'):
        super(TableRNNEncoder, self).__init__()
        self.split_type = split_type
        self.merge_type = merge_type
        self.hidden_size = encoder.hidden_size
        self.encoder = encoder
        if self.merge_type == 'mlp':
            self.merge = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.Tanh())

    def forward(self, tbl, tbl_len, tbl_split):
        """
        Encode table headers.
            :param tbl: header token list
            :param tbl_len: length of token list (num_table_header, batch)
            :param tbl_split: table header boundary list
        """
        tbl_context = encode_unsorted_batch(self.encoder, tbl, tbl_len)
        # --> (num_table_header, batch, hidden_size * num_directions)
        if self.split_type == 'outcell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand_as(tbl_split.data)
            enc_split = tbl_context[tbl_split.data, batch_index, :]
            enc_left, enc_right = enc_split[:-1], enc_split[1:]
        elif self.split_type == 'incell':
            batch_index = torch.LongTensor(range(tbl_split.data.size(1))).unsqueeze_(
                0).cuda().expand(tbl_split.data.size(0) - 1, tbl_split.data.size(1))
            split_left = (tbl_split.data[:-1] +
                          1).clamp(0, tbl_context.size(0) - 1)
            enc_left = tbl_context[split_left, batch_index, :]
            split_right = (tbl_split.data[1:] -
                           1).clamp(0, tbl_context.size(0) - 1)
            enc_right = tbl_context[split_right, batch_index, :]

        if self.merge_type == 'sub':
            return (enc_right - enc_left)
        elif self.merge_type == 'cat':
            # take half vector for each direction
            half_hidden_size = self.hidden_size // 2
            return torch.cat([enc_right[:, :, :half_hidden_size], enc_left[:, :, half_hidden_size:]], 2)
        elif self.merge_type == 'mlp':
            return self.merge(torch.cat([enc_right, enc_left], 2))


class MatchScorer(nn.Module):
    def __init__(self, input_size, score_size, dropout):
        super(MatchScorer, self).__init__()
        self.score_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, score_size),
            nn.Tanh(),
            nn.Linear(score_size, 1))
        self.log_sm = nn.LogSoftmax()

    def forward(self, q_enc, tbl_enc, tbl_mask):
        """
        Match and return table column score.
            :param q_enc: question encoding vectors (batch, rnn_size)
            :param tbl_enc: header encoding vectors (num_table_header, batch, rnn_size)
            :param tbl_num: length of token list
        """
        q_enc_expand = q_enc.unsqueeze(0).expand(
            tbl_enc.size(0), tbl_enc.size(1), q_enc.size(1))
        # (batch, num_table_header, input_size)
        feat = torch.cat((q_enc_expand, tbl_enc), 2).transpose(0, 1)
        # (batch, num_table_header)
        score = self.score_layer(feat).squeeze(2)
        # mask scores
        score_mask = score.masked_fill(tbl_mask, -float('inf'))
        # normalize
        return self.log_sm(score_mask)


class CondMatchScorer(nn.Module):
    def __init__(self, sel_match):
        super(CondMatchScorer, self).__init__()
        self.sel_match = sel_match

    def forward(self, cond_context_filter, tbl_enc, tbl_mask, emb_span_l=None):
        """
        Match and return table column score for cond decoder.
            :param cond_context: cond decoder's context vectors (num_cond*3, batch, rnn_size)
            :param tbl_enc: header encoding vectors (num_table_header, batch, rnn_size)
            :param tbl_num: length of token list
        """
        # -> (num_cond, batch, rnn_size)
        if emb_span_l is not None:
            # -> (num_cond, batch, 2*rnn_size)
            cond_context_filter = torch.cat(
                (cond_context_filter, emb_span_l), 2)
        r_list = []
        for cond_context_one in cond_context_filter:
            # -> (batch, num_table_header)
            r_list.append(self.sel_match(cond_context_one, tbl_enc, tbl_mask))
        # (num_cond, batch, num_table_header)
        return torch.stack(r_list, 0)


class CondDecoder(nn.Module):
    def __init__(self, rnn_type, bidirectional_encoder, num_layers, input_size, hidden_size, attn_type, attn_hidden, dropout, lock_dropout, weight_dropout):
        super(CondDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout)
        else:
            self.word_dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size, num_layers, dropout, weight_dropout)

        # Set up the standard attention.
        self.attn = table.modules.GlobalAttention(
            hidden_size, True, attn_type=attn_type, attn_hidden=attn_hidden)

    def forward(self, emb, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns = self._run_forward_pass(emb, context, state)

        # Update the state with the result.
        state.update_state(hidden)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        return RNNDecoderState(context, self.hidden_size, tuple([self._fix_enc_hidden(enc_hidden[i]) for i in range(len(enc_hidden))]))

    def _run_forward_pass(self, emb, context, state):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}

        if self.word_dropout is not None:
            emb = self.word_dropout(emb)

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state.hidden)

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)                   # (contxt_len, batch, d)
        )
        attns["std"] = attn_scores

        outputs = attn_outputs    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """

    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    @property
    def _all(self):
        return self.hidden

    def update_state(self, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        v_list = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                  for e in self._all]
        self.hidden = tuple(v_list)


class CoAttention(nn.Module):
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, dropout, weight_dropout, attn_type, attn_hidden):
        super(CoAttention, self).__init__()

        num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.no_pack_padded_seq = False

        self.rnn = _build_rnn(rnn_type, 2 * hidden_size, hidden_size //
                              num_directions, num_layers, dropout, weight_dropout, bidirectional)
        self.attn = table.modules.GlobalAttention(
            hidden_size, False, attn_type=attn_type, attn_hidden=attn_hidden)

    def forward(self, q_all, lengths, tbl_enc, tbl_mask):
        self.attn.applyMask(tbl_mask.data.unsqueeze(0))
        # attention
        emb, _ = self.attn(
            q_all.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            tbl_enc.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # feed to rnn
        if not isinstance(lengths, list):
            lengths = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, None)

        outputs = unpack(outputs)[0]

        return hidden_t, outputs


class ParserModel(nn.Module):
    def __init__(self, q_encoder, tbl_encoder, co_attention, agg_classifier, sel_match, lay_classifier, cond_embedding, lay_encoder, cond_decoder, cond_col_match, cond_span_l_match, cond_span_r_match, model_opt, pad_word_index):
        super(ParserModel, self).__init__()
        self.q_encoder = q_encoder
        self.tbl_encoder = tbl_encoder
        self.agg_classifier = agg_classifier
        self.sel_match = sel_match
        self.lay_classifier = lay_classifier
        self.cond_embedding = cond_embedding
        self.lay_encoder = lay_encoder
        self.cond_decoder = cond_decoder
        self.opt = model_opt
        self.span_merge = nn.Sequential(
            nn.Linear(2 * model_opt.rnn_size, model_opt.rnn_size),
            nn.Tanh())
        self.cond_col_match = cond_col_match
        self.cond_span_l_match = cond_span_l_match
        self.cond_span_r_match = cond_span_r_match
        self.pad_word_index = pad_word_index
        self.co_attention = co_attention

    def enc(self, q, q_len, ent, tbl, tbl_len, tbl_split, tbl_mask):
        q_enc, q_all = self.q_encoder(q, lengths=q_len, ent=ent)
        tbl_enc = self.tbl_encoder(tbl, tbl_len, tbl_split)
        if self.co_attention is not None:
            q_enc, q_all = self.co_attention(q_all, q_len, tbl_enc, tbl_mask)
        # (num_layers * num_directions, batch, hidden_size)
        q_ht, q_ct = q_enc
        batch_size = q_ht.size(1)
        q_ht = q_ht[-1] if not self.opt.brnn else q_ht[-2:].transpose(
            0, 1).contiguous().view(batch_size, -1)

        return q_enc, q_all, tbl_enc, q_ht, batch_size

    def select3(self, cond_context, start_index):
        return cond_context[start_index:cond_context.size(
            0):3]

    def forward(self, q, q_len, ent, tbl, tbl_len, tbl_split, tbl_mask, cond_op, cond_op_len, cond_col, cond_span_l, cond_span_r, lay):
        # encoding
        q_enc, q_all, tbl_enc, q_ht, batch_size = self.enc(
            q, q_len, ent, tbl, tbl_len, tbl_split, tbl_mask)

        # (1) decoding
        agg_out = self.agg_classifier(q_ht)
        sel_out = self.sel_match(q_ht, tbl_enc, tbl_mask)
        lay_out = self.lay_classifier(q_ht)

        # (2) decoding
        # emb_op
        if self.opt.layout_encode == 'rnn':
            emb_op = encode_unsorted_batch(
                self.lay_encoder, cond_op, cond_op_len.clamp(min=1))
        else:
            emb_op = self.cond_embedding(cond_op)
        # emb_col
        batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(
            0).cuda().expand(cond_col.size(0), cond_col.size(1))
        emb_col = tbl_enc[cond_col.data, batch_index, :]
        # emb_span_l/r: (num_cond, batch, hidden_size)
        emb_span_l = q_all[cond_span_l.data, batch_index, :]
        emb_span_r = q_all[cond_span_r.data, batch_index, :]
        emb_span = self.span_merge(torch.cat([emb_span_l, emb_span_r], 2))
        # stack embeddings
        # (seq_len*3, batch, hidden_size)
        emb = torch.stack([emb_op, emb_col, emb_span],
                          1).view(-1, batch_size, emb_op.size(2))
        # cond decoder
        self.cond_decoder.attn.applyMaskBySeqBatch(q)
        q_state = self.cond_decoder.init_decoder_state(q_all, q_enc)
        cond_context, _, _ = self.cond_decoder(emb, q_all, q_state)
        # cond col
        cond_context_0 = self.select3(cond_context, 0)
        cond_col_out = self.cond_col_match(cond_context_0, tbl_enc, tbl_mask)
        # cond span
        q_mask = Variable(q.data.eq(self.pad_word_index).transpose(
            0, 1), requires_grad=False)
        cond_context_1 = self.select3(cond_context, 1)
        cond_span_l_out = self.cond_span_l_match(
            cond_context_1, q_all, q_mask)
        cond_span_r_out = self.cond_span_r_match(
            cond_context_1, q_all, q_mask, emb_span_l)

        return agg_out, sel_out, lay_out, cond_col_out, cond_span_l_out, cond_span_r_out
