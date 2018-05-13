"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch.nn as nn
import torch.nn.functional as F

import table
import table.Models
import table.modules
from table.IO import TableDataset
from table.Models import ParserModel, RNNEncoder, CondDecoder, TableRNNEncoder, MatchScorer, CondMatchScorer, CoAttention
from lib.query import agg_ops, cond_ops
import torchtext.vocab
from table.modules.Embeddings import PartUpdateEmbedding


def make_word_embeddings(opt, word_dict, fields):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    emb_word = nn.Embedding(num_word, opt.word_vec_size,
                            padding_idx=word_padding_idx)

    if len(opt.pre_word_vecs) > 0:
        vectors = torchtext.vocab.GloVe(
            name="840B", cache=opt.pre_word_vecs, dim=str(opt.word_vec_size))
        fields["src"].vocab.load_vectors(vectors)
        emb_word.weight.data.copy_(fields["src"].vocab.vectors)

    if opt.fix_word_vecs:
        # <unk> is 0
        num_special = len(table.IO.special_token_list)
        # zero vectors in the fixed embedding (emb_word)
        emb_word.weight.data[:num_special].zero_()
        emb_special = nn.Embedding(
            num_special, opt.word_vec_size, padding_idx=word_padding_idx)
        emb = PartUpdateEmbedding(num_special, emb_special, emb_word)
        return emb
    else:
        return emb_word


def make_embeddings(word_dict, vec_size):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    w_embeddings = nn.Embedding(
        num_word, vec_size, padding_idx=word_padding_idx)
    return w_embeddings


def make_encoder(opt, embeddings, ent_embedding=None):
    # "rnn" or "brnn"
    return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.dropout, opt.lock_dropout, opt.weight_dropout, embeddings, ent_embedding)


def make_table_encoder(opt, embeddings):
    # "rnn" or "brnn"
    return TableRNNEncoder(make_encoder(opt, embeddings), opt.split_type, opt.merge_type)


def make_cond_decoder(opt):
    input_size = opt.rnn_size
    return CondDecoder(opt.rnn_type, opt.brnn, opt.dec_layers, input_size, opt.rnn_size, opt.global_attention, opt.attn_hidden, opt.dropout, opt.lock_dropout, opt.weight_dropout)


def make_co_attention(opt):
    if opt.co_attention:
        return CoAttention(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.dropout, opt.weight_dropout, opt.global_attention, opt.attn_hidden)
    return None


def make_base_model(model_opt, fields, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # embedding
    w_embeddings = make_word_embeddings(model_opt, fields["src"].vocab, fields)

    if model_opt.ent_vec_size > 0:
        ent_embedding = make_embeddings(
            fields["ent"].vocab, model_opt.ent_vec_size)
    else:
        ent_embedding = None

    # Make question encoder.
    q_encoder = make_encoder(model_opt, w_embeddings, ent_embedding)
    # Make table encoder.
    tbl_encoder = make_table_encoder(model_opt, w_embeddings)

    co_attention = make_co_attention(model_opt)

    agg_classifier = nn.Sequential(
        nn.Dropout(model_opt.dropout),
        nn.Linear(model_opt.rnn_size, len(agg_ops)),
        nn.LogSoftmax())
    sel_match = MatchScorer(2 * model_opt.rnn_size,
                            model_opt.score_size, model_opt.dropout)
    lay_classifier = nn.Sequential(
        nn.Dropout(model_opt.dropout),
        nn.Linear(model_opt.rnn_size, len(fields['lay'].vocab)),
        nn.LogSoftmax())

    # embedding
    # layout encoding
    if model_opt.layout_encode == 'rnn':
        cond_embedding = make_embeddings(
            fields["cond_op"].vocab, model_opt.cond_op_vec_size)
        lay_encoder = make_encoder(model_opt, cond_embedding)
    else:
        cond_embedding = make_embeddings(
            fields["cond_op"].vocab, model_opt.rnn_size)
        lay_encoder = None

    # Make cond models.
    cond_decoder = make_cond_decoder(model_opt)
    cond_col_match = CondMatchScorer(
        MatchScorer(2 * model_opt.rnn_size, model_opt.score_size, model_opt.dropout))
    cond_span_l_match = CondMatchScorer(
        MatchScorer(2 * model_opt.rnn_size, model_opt.score_size, model_opt.dropout))
    cond_span_r_match = CondMatchScorer(
        MatchScorer(3 * model_opt.rnn_size, model_opt.score_size, model_opt.dropout))

    # Make ParserModel
    pad_word_index = fields["src"].vocab.stoi[table.IO.PAD_WORD]
    model = ParserModel(q_encoder, tbl_encoder, co_attention, agg_classifier, sel_match, lay_classifier, cond_embedding,
                        lay_encoder, cond_decoder, cond_col_match, cond_span_l_match, cond_span_r_match, model_opt, pad_word_index)

    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])

    model.cuda()

    return model
