"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch.nn as nn
import torch.nn.functional as F

import table
import table.Models
import table.modules
from table.Models import ParserModel, RNNEncoder, SeqDecoder, SeqDecoderParentFeedInput, LayCoAttention, QCoAttention
import torchtext.vocab
from table.modules.Embeddings import PartUpdateEmbedding


def make_word_embeddings(opt, word_dict, fields):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    emb_word = nn.Embedding(num_word, opt.word_vec_size,
                            padding_idx=word_padding_idx)

    if len(opt.pre_word_vecs) > 0:
        if opt.word_vec_size == 150:
            dim_list = ['100', '50']
        elif opt.word_vec_size == 250:
            dim_list = ['200', '50']
        else:
            dim_list = [str(opt.word_vec_size), ]
        vectors = [torchtext.vocab.GloVe(
            name="6B", cache=opt.pre_word_vecs, dim=it) for it in dim_list]
        word_dict.load_vectors(vectors)
        emb_word.weight.data.copy_(word_dict.vectors)

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
    return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_enc, opt.weight_dropout, embeddings, ent_embedding)


def make_layout_encoder(opt, embeddings):
    return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers, opt.decoder_input_size, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_enc, opt.weight_dropout, embeddings, None)


def make_q_co_attention(opt):
    if opt.q_co_attention:
        return QCoAttention(opt.rnn_type, opt.brnn, opt.enc_layers, opt.rnn_size, opt.decoder_input_size, opt.dropout, opt.weight_dropout, 'dot', opt.attn_hidden)
    return None


def make_lay_co_attention(opt):
    if opt.lay_co_attention:
        return LayCoAttention(opt.rnn_type, opt.brnn, opt.enc_layers, opt.decoder_input_size, opt.rnn_size, opt.dropout, opt.weight_dropout, 'mlp', opt.attn_hidden)
    return None


def make_decoder(opt, fields, field_name, embeddings, input_size):
    if opt.parent_feed == 'input':
        input_add = opt.parent_feed_hidden if opt.parent_feed_hidden > 0 else opt.rnn_size
        decoder = SeqDecoderParentFeedInput(opt.rnn_type, opt.brnn, opt.dec_layers, embeddings, input_size + input_add, opt.rnn_size,
                                            opt.global_attention, opt.attn_hidden, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_dec, opt.weight_dropout, opt.parent_feed_hidden)
    else:
        decoder = SeqDecoder(opt.rnn_type, opt.brnn, opt.dec_layers, embeddings, input_size, opt.rnn_size,
                             opt.global_attention, opt.attn_hidden, opt.dropout, opt.dropout_i, opt.lock_dropout, opt.dropword_dec, opt.weight_dropout)
    if opt.parent_feed == 'output':
        size_classifier = 2 * opt.rnn_size
    else:
        size_classifier = opt.rnn_size
    classifier = nn.Sequential(
        nn.Dropout(opt.dropout),
        nn.Linear(size_classifier, len(fields[field_name].vocab)),
        nn.LogSoftmax())
    return decoder, classifier


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
    if model_opt.seprate_encoder:
        q_tgt_encoder = make_encoder(model_opt, w_embeddings, ent_embedding)
        q_encoder = (q_encoder, q_tgt_encoder)

    if model_opt.layout_token_prune:
        w_token_embeddings = make_word_embeddings(
            model_opt, fields["src"].vocab, fields)
        q_token_encoder = make_encoder(
            model_opt, w_token_embeddings, ent_embedding)
        token_pruner = nn.Sequential(
            nn.Dropout(model_opt.dropout),
            # skip special tokens
            nn.Linear(model_opt.rnn_size, len(fields['lay'].vocab) - len(table.IO.special_token_list)))
    else:
        q_token_encoder = None
        token_pruner = None

    # Make layout decoder models.
    if model_opt.bpe:
        lay_field = 'lay_bpe'
    else:
        lay_field = 'lay'
    lay_embeddings = make_embeddings(
        fields[lay_field].vocab, model_opt.decoder_input_size)
    lay_decoder, lay_classifier = make_decoder(
        model_opt, fields, lay_field, lay_embeddings, model_opt.decoder_input_size)

    # Make target decoder models.
    if model_opt.no_share_emb_layout_encoder:
        lay_encoder_embeddings = make_embeddings(
            fields[lay_field].vocab, model_opt.decoder_input_size)
    else:
        lay_encoder_embeddings = lay_embeddings
    if model_opt.no_lay_encoder:
        lay_encoder = lay_embeddings
    else:
        lay_encoder = make_layout_encoder(model_opt, lay_encoder_embeddings)

    q_co_attention = make_q_co_attention(model_opt)
    lay_co_attention = make_lay_co_attention(model_opt)

    tgt_embeddings = make_embeddings(
        fields['tgt'].vocab, model_opt.decoder_input_size)
    tgt_decoder, tgt_classifier = make_decoder(
        model_opt, fields, 'tgt', None, model_opt.decoder_input_size)

    # Make ParserModel
    model = ParserModel(q_encoder, q_token_encoder, token_pruner, lay_decoder, lay_classifier,
                        lay_encoder, q_co_attention,lay_co_attention, tgt_embeddings, tgt_decoder, tgt_classifier, model_opt)

    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])

    model.cuda()

    return model
