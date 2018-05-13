# -*- coding: utf-8 -*-

import codecs
import json
import random as rnd
import numpy as np
from collections import Counter, defaultdict
from itertools import chain, count
from six import string_types

import torch
import torchtext.data
import torchtext.vocab
from tree import SCode

UNK_WORD = '<unk>'
UNK = 0
PAD_WORD = '<blank>'
PAD = 1
BOS_WORD = '<s>'
BOS = 2
EOS_WORD = '</s>'
EOS = 3
SKP_WORD = '<sk>'
SKP = 4
RIG_WORD = '<]>'
RIG = 5
LFT_WORD = '<[>'
LFT = 6
special_token_list = [UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, SKP_WORD, RIG_WORD, LFT_WORD]


def get_parent_index(tk_list):
    stack = [0]
    r_list = []
    for i, tk in enumerate(tk_list):
        r_list.append(stack[-1])
        if tk.startswith('('):
            # +1: because the parent of the top level is 0
            stack.append(i+1)
        elif tk ==')':
            stack.pop()
    # for EOS (</s>)
    r_list.append(0)
    return r_list


def get_tgt_mask(lay_skip):
    # 0: use layout encoding vectors; 1: use target word embeddings;
    # with a <s> token at the first position
    return [1] + [1 if tk in (SKP_WORD, RIG_WORD) else 0 for tk in lay_skip]


def get_lay_index(lay_skip):
    # with a <s> token at the first position
    r_list = [0]
    k = 0
    for tk in lay_skip:
        if tk in (SKP_WORD, RIG_WORD):
            r_list.append(0)
        else:
            r_list.append(k)
            k += 1
    return r_list


def get_tgt_loss(line, mask_target_loss):
    r_list = []
    for tk_tgt, tk_lay_skip in zip(line['tgt'], line['lay_skip']):
        if tk_lay_skip in (SKP_WORD, RIG_WORD):
            r_list.append(tk_tgt)
        else:
            if mask_target_loss:
                r_list.append(PAD_WORD)
            else:
                r_list.append(tk_tgt)
    return r_list


def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def filter_counter(freqs, min_freq):
    cnt = Counter()
    for k, v in freqs.items():
        if (min_freq is None) or (v >= min_freq):
            cnt[k] = v
    return cnt


def merge_vocabs(vocabs, min_freq=0, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = Counter()
    for vocab in vocabs:
        merged += filter_counter(vocab.freqs, min_freq)
    return torchtext.vocab.Vocab(merged,
                                 specials=list(special_token_list),
                                 max_size=vocab_size, min_freq=min_freq)


def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def _preprocess_json(js, opt):
    t = SCode((js['token'], js['type']))
    js['lay'] = t.layout(add_skip=False)
    js['lay_skip'] = t.layout(add_skip=True)
    assert len(t.target()) == len(js['lay_skip']), (list(zip(t.target(), js['lay_skip'])), ' '.join(js['tgt']))
    js['tgt'] = t.target()

def read_anno_json(anno_path, opt):
    with codecs.open(anno_path, "r", "utf-8") as corpus_file:
        js_list = [json.loads(line) for line in corpus_file]
        for js in js_list:
            _preprocess_json(js, opt)
    return js_list


class TableDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)

    def __init__(self, anno, fields, permute_order, opt, filter_ex, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        anno: location of annotated data / js_list
        filter_ex: False - keep all the examples for evaluation (should not have filtered examples); True - filter examples with unmatched spans;
        """
        if isinstance(anno, string_types):
            js_list = read_anno_json(anno, opt)
        else:
            js_list = anno

        src_data = self._read_annotated_file(opt, js_list, 'src', filter_ex)
        src_examples = self._construct_examples(src_data, 'src')

        lay_data = self._read_annotated_file(opt, js_list, 'lay', filter_ex)
        lay_examples = self._construct_examples(lay_data, 'lay')

        # without <s> and </s>
        lay_e_data = self._read_annotated_file(opt, js_list, 'lay', filter_ex)
        lay_e_examples = self._construct_examples(lay_e_data, 'lay_e')

        lay_index_data = self._read_annotated_file(
            opt, js_list, 'lay_index', filter_ex)
        lay_index_examples = self._construct_examples(
            lay_index_data, 'lay_index')

        lay_parent_index_data = self._read_annotated_file(
            opt, js_list, 'lay_parent_index', filter_ex)
        lay_parent_index_examples = self._construct_examples(
            lay_parent_index_data, 'lay_parent_index')

        copy_to_tgt_data = self._read_annotated_file(
            opt, js_list, 'copy_to_tgt', filter_ex)
        copy_to_tgt_examples = self._construct_examples(
            copy_to_tgt_data, 'copy_to_tgt')

        copy_to_ext_data = self._read_annotated_file(
            opt, js_list, 'copy_to_ext', filter_ex)
        copy_to_ext_examples = self._construct_examples(
            copy_to_ext_data, 'copy_to_ext')

        tgt_mask_data = self._read_annotated_file(
            opt, js_list, 'tgt_mask', filter_ex)
        tgt_mask_examples = self._construct_examples(tgt_mask_data, 'tgt_mask')

        tgt_data = self._read_annotated_file(opt, js_list, 'tgt', filter_ex)
        tgt_examples = self._construct_examples(tgt_data, 'tgt')

        tgt_parent_index_data = self._read_annotated_file(
            opt, js_list, 'tgt_parent_index', filter_ex)
        tgt_parent_index_examples = self._construct_examples(
            tgt_parent_index_data, 'tgt_parent_index')

        tgt_loss_data = self._read_annotated_file(
            opt, js_list, 'tgt_loss', filter_ex)
        tgt_loss_examples = self._construct_examples(tgt_loss_data, 'tgt_loss')

        tgt_copy_ext_data = self._read_annotated_file(
            opt, js_list, 'tgt_copy_ext', filter_ex)
        tgt_copy_ext_examples = self._construct_examples(tgt_copy_ext_data, 'tgt_copy_ext')

        # examples: one for each src line or (src, tgt) line pair.
        examples = [join_dicts(*it) for it in zip(src_examples, lay_examples, lay_e_examples, lay_index_examples, lay_parent_index_examples, copy_to_tgt_examples, copy_to_ext_examples, tgt_mask_examples, tgt_examples, tgt_parent_index_examples, tgt_loss_examples, tgt_copy_ext_examples)]
        # the examples should not contain None
        len_before_filter = len(examples)
        examples = list(filter(lambda x: all(
            (v is not None for k, v in x.items())), examples))
        len_after_filter = len(examples)
        num_filter = len_before_filter - len_after_filter
        if num_filter > 0:
            print('Filter #examples (with None): {} / {} = {:.2%}'.format(num_filter,
                                                                          len_before_filter, num_filter / len_before_filter))

        # Peek at the first to see which fields are used.
        ex = examples[0]
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        def construct_final(examples):
            for i, ex in enumerate(examples):
                yield torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields)

        def filter_pred(example):
            return True

        super(TableDataset, self).__init__(
            construct_final(examples), fields, filter_pred)

    def _read_annotated_file(self, opt, js_list, field, filter_ex):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)
        """
        if field in ('src', 'lay'):
            lines = (line[field] for line in js_list)
        elif field in ('copy_to_tgt','copy_to_ext'):
            lines = (line['src'] for line in js_list)
        elif field in ('tgt',):
            def _tgt(line):
                r_list = []
                for tk_tgt, tk_lay_skip in zip(line['tgt'], line['lay_skip']):
                    if tk_lay_skip in (SKP_WORD, RIG_WORD):
                        r_list.append(tk_tgt)
                    else:
                        r_list.append(PAD_WORD)
                return r_list
            lines = (_tgt(line) for line in js_list)
        elif field in ('tgt_copy_ext',):
            def _tgt_copy_ext(line):
                r_list = []
                src_set = set(line['src'])
                for tk_tgt in line['tgt']:
                    if tk_tgt in src_set:
                        r_list.append(tk_tgt)
                    else:
                        r_list.append(UNK_WORD)
                return r_list
            lines = (_tgt_copy_ext(line) for line in js_list)
        elif field in ('tgt_loss',):
            lines = (get_tgt_loss(line, False) for line in js_list)
        elif field in ('tgt_mask',):
            lines = (get_tgt_mask(line['lay_skip']) for line in js_list)
        elif field in ('lay_index',):
            lines = (get_lay_index(line['lay_skip']) for line in js_list)
        elif field in ('lay_parent_index',):
            lines = (get_parent_index(line['lay']) for line in js_list)
        elif field in ('tgt_parent_index',):
            lines = (get_parent_index(line['tgt']) for line in js_list)
        else:
            raise NotImplementedError
        for line in lines:
            yield line

    def _construct_examples(self, lines, side):
        for words in lines:
            example_dict = {side: words}
            yield example_dict

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(TableDataset, self).__reduce_ex__()

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = TableDataset.get_fields()
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def get_fields():
        fields = {}
        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD, include_lengths=True)
        fields["lay"] = torchtext.data.Field(
            init_token=BOS_WORD, include_lengths=True, eos_token=EOS_WORD, pad_token=PAD_WORD)
        fields["lay_e"] = torchtext.data.Field(
            include_lengths=False, pad_token=PAD_WORD)
        fields["lay_index"] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["lay_parent_index"] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["copy_to_tgt"] = torchtext.data.Field(pad_token=UNK_WORD)
        fields["copy_to_ext"] = torchtext.data.Field(pad_token=UNK_WORD)
        fields["tgt_mask"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor, pad_token=1)
        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)
        fields["tgt_copy_ext"] = torchtext.data.Field(
            init_token=UNK_WORD, eos_token=UNK_WORD, pad_token=UNK_WORD)
        fields["tgt_parent_index"] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["tgt_loss"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)
        fields["indices"] = torchtext.data.Field(
            use_vocab=False, sequential=False)
        return fields

    @staticmethod
    def build_vocab(train, dev, test, opt):
        fields = train.fields

        src_vocab_all = []
        # build vocabulary only based on the training set
        # the last one should be the variable 'train'
        for split in (dev, test, train,):
            fields['src'].build_vocab(split, min_freq=0)
            src_vocab_all.extend(list(fields['src'].vocab.stoi.keys()))

        # build vocabulary only based on the training set
        for field_name in ('src', 'lay', 'lay_e'):
            fields[field_name].build_vocab(
                train, min_freq=opt.src_words_min_frequency)

        # build vocabulary only based on the training set
        for field_name in ('tgt', 'tgt_loss'):
            fields[field_name].build_vocab(
                train, min_freq=opt.tgt_words_min_frequency)

        for field_name in ('src', 'lay', 'lay_e'):
            fields[field_name].vocab = merge_vocabs(
                [fields[field_name].vocab], min_freq=opt.src_words_min_frequency)

        tgt_merge_name_list = ['tgt', 'tgt_loss']
        tgt_merge = merge_vocabs([fields[field_name].vocab for field_name in tgt_merge_name_list], min_freq=opt.tgt_words_min_frequency)
        for field_name in tgt_merge_name_list:
            fields[field_name].vocab = tgt_merge
        
        fields['copy_to_tgt'].vocab = fields['tgt'].vocab
        # fields['src_all'].vocab - fields['tgt'].vocab
        cnt_ext = Counter()
        for k in src_vocab_all:
            if k not in fields['tgt'].vocab.stoi:
                cnt_ext[k] = 1
        fields['copy_to_ext'].vocab = torchtext.vocab.Vocab(cnt_ext, specials=list(special_token_list), min_freq=0)
        fields['tgt_copy_ext'].vocab = fields['copy_to_ext'].vocab
