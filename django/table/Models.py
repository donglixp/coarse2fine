from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F

import table
from table.Utils import aeq, sort_for_pack
from table.modules.embed_regularize import embedded_dropout
from table.modules.cross_entropy_smooth import onehot
from table.Utils import argmax


def _build_rnn(rnn_type, input_size, hidden_size, num_layers, dropout, weight_dropout, bidirectional=False):
    rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    if weight_dropout > 0:
        param_list = ['weight_hh_l' + str(i) for i in range(num_layers)]
        if bidirectional:
            param_list += [it + '_reverse' for it in param_list]
        rnn = table.modules.WeightDrop(rnn, param_list, dropout=weight_dropout)
    return rnn


class RNNEncoder(nn.Module):
    """ The standard RNN encoder. """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, dropout_i, lock_dropout, dropword, weight_dropout, embeddings, ent_embedding):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.ent_embedding = ent_embedding
        self.no_pack_padded_seq = False
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout_i)
        else:
            self.word_dropout = nn.Dropout(dropout_i)
        self.dropword = dropword

        # Use pytorch version when available.
        input_size = embeddings.embedding_dim
        if ent_embedding is not None:
            input_size += ent_embedding.embedding_dim
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size // num_directions, num_layers, dropout, weight_dropout, bidirectional)

    def forward(self, input, lengths=None, hidden=None, ent=None):
        if self.training and (self.dropword > 0):
            emb = embedded_dropout(
                self.embeddings, input, dropout=self.dropword)
        else:
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


class SeqDecoder(nn.Module):
    def __init__(self, rnn_type, bidirectional_encoder, num_layers, embeddings, input_size, hidden_size, attn_type, attn_hidden, dropout, dropout_i, lock_dropout, dropword, weight_dropout):
        super(SeqDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.input_size = input_size
        self.hidden_size = hidden_size
        if lock_dropout:
            self.word_dropout = table.modules.LockedDropout(dropout_i)
        else:
            self.word_dropout = nn.Dropout(dropout_i)
        self.dropword = dropword

        # Build the RNN.
        self.rnn = _build_rnn(rnn_type, input_size,
                              hidden_size, num_layers, dropout, weight_dropout)

        # Set up the standard attention.
        self.attn = table.modules.GlobalAttention(
            hidden_size, True, attn_type=attn_type, attn_hidden=attn_hidden)

    def forward(self, inp, context, state, parent_index):
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

        if self.embeddings is not None:
            if self.training and (self.dropword > 0):
                emb = embedded_dropout(
                    self.embeddings, inp, dropout=self.dropword)
            else:
                emb = self.embeddings(inp)
        else:
            emb = inp
        if self.word_dropout is not None:
            emb = self.word_dropout(emb)

        # Run the forward pass of the RNN.
        hidden, outputs, attns, rnn_output, concat_c = self._run_forward_pass(
            emb, context, state, parent_index)

        # Update the state with the result.
        state.update_state(hidden)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        attns = torch.stack(attns)

        return outputs, state, attns, rnn_output, concat_c

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

    def _run_forward_pass(self, emb, context, state, parent_index):
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

        if self.word_dropout is not None:
            emb = self.word_dropout(emb)

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state.hidden)

        # Calculate the attention.
        attn_outputs, attn_scores, concat_c = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)                   # (contxt_len, batch, d)
        )

        outputs = attn_outputs    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attn_scores, rnn_output, concat_c


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
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, context_size, dropout, weight_dropout, attn_type, attn_hidden):
        super(CoAttention, self).__init__()

        if (hidden_size != context_size) and (attn_type != 'mlp'):
            self.linear_context = nn.Linear(
                context_size, hidden_size, bias=False)
            context_size = hidden_size
        else:
            self.linear_context = None

        num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.no_pack_padded_seq = False

        self.rnn = _build_rnn(rnn_type, hidden_size + context_size, hidden_size //
                              num_directions, num_layers, dropout, weight_dropout, bidirectional)
        self.attn = table.modules.GlobalAttention(
            hidden_size, False, attn_type=attn_type, attn_hidden=attn_hidden, context_size=context_size)


class QCoAttention(CoAttention):
    def forward(self, q_all, lengths, lay_all, lay):
        self.attn.applyMaskBySeqBatch(lay)
        if self.linear_context is not None:
            lay_all = self.linear_context(lay_all)
        # attention
        emb, _,_ = self.attn(
            q_all.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            lay_all.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # feed to rnn
        if not isinstance(lengths, list):
            lengths = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, None)

        outputs = unpack(outputs)[0]

        return hidden_t, outputs


class LayCoAttention(CoAttention):
    def run_rnn_unsorted_batch(self, emb, lengths):
        # sort for pack()
        idx_sorted, tbl_len_sorted, idx_map_back = sort_for_pack(lengths)
        tbl_sorted = emb.index_select(1, Variable(
            torch.LongTensor(idx_sorted).cuda(), requires_grad=False))
        # tbl_context: (seq_len, batch, hidden_size * num_directions)
        packed_emb = pack(tbl_sorted, tbl_len_sorted)
        tbl_context, __ = self.rnn(packed_emb, None)
        tbl_context = unpack(tbl_context)[0]
        # recover the sort for pack()
        v_idx_map_back = Variable(torch.LongTensor(
            idx_map_back).cuda(), requires_grad=False)
        tbl_context = tbl_context.index_select(1, v_idx_map_back)
        return tbl_context

    def forward(self, lay_all, lengths, q_all, q):
        self.attn.applyMaskBySeqBatch(q)
        if self.linear_context is not None:
            q_all = self.linear_context(q_all)
        # attention
        emb, _,_ = self.attn(
            lay_all.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            q_all.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # feed to rnn
        outputs = self.run_rnn_unsorted_batch(emb, lengths)

        return outputs


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source. For each source sentence we have a `src_map` that maps each source word to an index in `tgt_dict` if it known, or else to an extra word. The copy generator is an extended version of the standard generator that computse three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead. taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary, computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    Args:
       hidden_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """

    def __init__(self, dropout, hidden_size, context_size, tgt_dict, ext_dict, copy_prb):
        super(CopyGenerator, self).__init__()
        self.copy_prb = copy_prb
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, len(tgt_dict))
        if copy_prb == 'hidden':
            self.linear_copy = nn.Linear(hidden_size, 1)
        elif copy_prb == 'hidden_context':
            self.linear_copy = nn.Linear(hidden_size + context_size, 1)
        else:
            raise NotImplementedError
        self.tgt_dict = tgt_dict
        self.ext_dict = ext_dict

    def forward(self, hidden, dec_rnn_output, concat_c, attn, copy_to_ext, copy_to_tgt):
        """
        Compute a distribution over the target dictionary extended by the dynamic dictionary implied by compying source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[tlen * batch, hidden_size]`
           attn (`FloatTensor`): attn for each `[tlen * batch, src_len]`
           copy_to_ext (`FloatTensor`): A sparse indicator matrix mapping each source word to its index in the "extended" vocab containing. `[src_len, batch]`
           copy_to_tgt (`FloatTensor`): A sparse indicator matrix mapping each source word to its index in the target vocab containing. `[src_len, batch]`
        """
        dec_seq_len = hidden.size(0)
        batch_size = hidden.size(1)
        # -> (targetL_ * batch_, rnn_size)
        hidden = hidden.view(dec_seq_len * batch_size, -1)
        dec_rnn_output = dec_rnn_output.view(dec_seq_len * batch_size, -1)
        concat_c = concat_c.view(dec_seq_len * batch_size, -1)
        # -> (targetL_ * batch_, sourceL_)
        attn = attn.view(dec_seq_len * batch_size, -1)

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch = copy_to_ext.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        hidden = self.dropout(hidden)

        # Original probabilities.
        logits = self.linear(hidden)
        # logits[:, self.tgt_dict.stoi[table.IO.PAD_WORD]] = -float('inf')
        prob_log = F.log_softmax(logits)
        # return prob_log.view(dec_seq_len, batch_size, -1)

        # Probability of copying p(z=1) batch.
        # copy = F.sigmoid(self.linear_copy(hidden))
        if self.copy_prb == 'hidden':
            copy = F.sigmoid(self.linear_copy(dec_rnn_output))
        elif self.copy_prb == 'hidden_context':
            copy = F.sigmoid(self.linear_copy(concat_c))
        else:
            raise NotImplementedError

        def safe_log(v):
            return torch.log(v.clamp(1e-3, 1 - 1e-3))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob_log = prob_log + safe_log(copy).expand_as(prob_log)
        mul_attn = torch.mul(attn, 1.0 - copy.expand_as(attn))
        # copy to extend vocabulary
        copy_to_ext_onehot = onehot(
            copy_to_ext, N=len(self.ext_dict), ignore_index=self.ext_dict.stoi[table.IO.UNK_WORD]).float()
        ext_copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1),
                                  copy_to_ext_onehot.transpose(0, 1)).transpose(0, 1).contiguous().view(-1, len(self.ext_dict))
        ext_copy_prob_log = safe_log(ext_copy_prob)

        return torch.cat([prob_log, ext_copy_prob_log], 1).view(dec_seq_len, batch_size, -1)

        # copy to target vocabulary
        copy_to_tgt_onehot = onehot(
            copy_to_tgt, N=len(self.tgt_dict), ignore_index=self.tgt_dict.stoi[table.IO.UNK_WORD]).float()
        tgt_add_copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1),
                                      copy_to_tgt_onehot.transpose(0, 1)).transpose(0, 1).contiguous().view(-1, len(self.tgt_dict))
        out_prob = torch.exp(out_prob_log) + tgt_add_copy_prob

        return torch.log(torch.cat([out_prob, ext_copy_prob], 1)).view(dec_seq_len, batch_size, -1)


class ParserModel(nn.Module):
    def __init__(self, q_encoder, q_token_encoder, token_pruner, lay_decoder, lay_classifier, lay_encoder, q_co_attention, lay_co_attention, tgt_embeddings, tgt_decoder, tgt_classifier, model_opt):
        super(ParserModel, self).__init__()
        if model_opt.seprate_encoder:
            self.q_encoder, self.q_tgt_encoder = q_encoder
        else:
            self.q_encoder = q_encoder
        self.q_token_encoder = q_token_encoder
        self.token_pruner = token_pruner
        self.lay_decoder = lay_decoder
        self.lay_classifier = lay_classifier
        self.lay_encoder = lay_encoder
        self.q_co_attention = q_co_attention
        self.lay_co_attention = lay_co_attention
        self.tgt_embeddings = tgt_embeddings
        self.tgt_decoder = tgt_decoder
        self.tgt_classifier = tgt_classifier
        self.opt = model_opt

    def enc_to_ht(self, q_enc, batch_size):
        # (num_layers * num_directions, batch, hidden_size)
        q_ht, q_ct = q_enc
        q_ht = q_ht[-1] if not self.opt.brnn else q_ht[-2:].transpose(
            0, 1).contiguous().view(batch_size, -1).unsqueeze(0)
        return q_ht

    def run_decoder(self, decoder, classifier, q, q_all, q_enc, inp, parent_index):
        batch_size = q.size(1)
        decoder.attn.applyMaskBySeqBatch(q)
        q_state = decoder.init_decoder_state(q_all, q_enc)
        dec_all, _, attn_scores, _, _ = decoder(
            inp, q_all, q_state, parent_index)
        dec_seq_len = dec_all.size(0)
        dec_all = dec_all.view(dec_seq_len * batch_size, -1)
        dec_out = classifier(dec_all)
        dec_out = dec_out.view(dec_seq_len, batch_size, -1)
        return dec_out, attn_scores

    def run_copy_decoder(self, decoder, classifier, q, q_all, q_enc, inp, parent_index, copy_to_ext, copy_to_tgt):
        batch_size = q.size(1)
        decoder.attn.applyMaskBySeqBatch(q)
        q_state = decoder.init_decoder_state(q_all, q_enc)
        dec_all, _, attn_scores, dec_rnn_output, concat_c = decoder(
            inp, q_all, q_state, parent_index)
        dec_out = classifier(dec_all, dec_rnn_output,
                             concat_c, attn_scores, copy_to_ext, copy_to_tgt)
        return dec_out, attn_scores

    def forward(self, q, q_len, ent, lay, lay_e, lay_len, lay_index, tgt_mask, tgt, lay_parent_index, tgt_parent_index, copy_to_ext, copy_to_tgt):
        batch_size = q.size(1)
        # encoding
        q_enc, q_all = self.q_encoder(q, lengths=q_len, ent=ent)
        if self.opt.seprate_encoder:
            q_tgt_enc, q_tgt_all = self.q_tgt_encoder(
                q, lengths=q_len, ent=ent)
        else:
            q_tgt_enc, q_tgt_all = q_enc, q_all

        if self.token_pruner:
            q_token_enc, __ = self.q_token_encoder(q, lengths=q_len, ent=ent)
            # (num_layers * num_directions, batch, hidden_size)
            q_token_ht, __ = q_token_enc
            batch_size = q_token_ht.size(1)
            q_token_ht = q_token_ht[-1] if not self.opt.brnn else q_token_ht[-2:].transpose(
                0, 1).contiguous().view(batch_size, -1)
            token_out = self.token_pruner(q_token_ht).t()
        else:
            token_out = None

        # layout decoding
        dec_in_lay = lay[:-1]
        lay_out, lay_attn_scores = self.run_decoder(
            self.lay_decoder, self.lay_classifier, q, q_all, q_enc, dec_in_lay, lay_parent_index)

        if self.opt.coverage_loss > 0:
            q_mask = Variable(q.data.ne(table.IO.PAD).t(), requires_grad=False)
            # targetL_, batch_, sourceL_ = lay_attn_scores.size()
            coverage_T = lay_attn_scores.mean(0, keepdim=False)
            loss_coverage = (0.7 / Variable(lay_len.unsqueeze(1).float(),
                                            requires_grad=False) - coverage_T).clamp(min=0).masked_select(q_mask).sum()
        else:
            loss_coverage = None

        # layout encoding
        # data used for layout encoding
        lay_e_len = lay_len - 2
        # (lay_len, batch, lay_size)
        if self.opt.no_lay_encoder:
            lay_all = self.lay_encoder(lay_e)
        else:
            lay_all = encode_unsorted_batch(
                self.lay_encoder, lay_e, lay_e_len)
        # co-attention
        if self.lay_co_attention is not None:
            lay_all = self.lay_co_attention(lay_all, lay_e_len, q_all, q)

        # target decoding
        batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(
            0).cuda().expand(lay_index.size(0), lay_index.size(1))
        # (tgt_len, batch, lay_size)
        lay_select = lay_all[lay_index.data, batch_index, :]
        # (tgt_len, batch, lay_size)
        tgt_inp_emb = self.tgt_embeddings(tgt[:-1])
        # (tgt_len, batch) -> (tgt_len, batch, lay_size)
        tgt_mask_expand = tgt_mask.unsqueeze(2).expand_as(tgt_inp_emb)
        dec_inp = tgt_inp_emb.mul(tgt_mask_expand) + \
            lay_select.mul(1 - tgt_mask_expand)

        # co-attention
        if self.q_co_attention is not None:
            q_tgt_enc_co_attn, q_tgt_all_co_attn = self.q_co_attention(
                q_tgt_all, q_len, lay_all, lay_e)
            # q_tgt_enc = tuple(((it0 + it1) * 0.5 for it0, it1 in zip(q_tgt_enc, q_tgt_enc_co_attn)))
            # q_tgt_all = (q_tgt_all + q_tgt_all_co_attn) * 0.5
            q_tgt_enc, q_tgt_all = q_tgt_enc_co_attn, q_tgt_all_co_attn

        tgt_out, __ = self.run_copy_decoder(
            self.tgt_decoder, self.tgt_classifier, q, q_tgt_all, q_tgt_enc, dec_inp, tgt_parent_index, copy_to_ext, copy_to_tgt)

        return lay_out, tgt_out, token_out, loss_coverage
