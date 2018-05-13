import json
from collections import defaultdict

from tree import STree, is_tree_eq


class ParseResult(object):
    def __init__(self, idx, lay, tgt, token_prune):
        self.idx = idx
        self.lay = lay
        self.tgt = tgt
        self.token_prune = token_prune
        self.correct = defaultdict(lambda: 0)
        self.incorrect_prune = set()

    def eval(self, gold):
        if is_tree_eq(self.lay, gold['lay'], not_layout=False):
            self.correct['lay'] = 1
        # else:
        #     print(' '.join(gold['src']))
        #     print('pred:', self.lay)
        #     print('gold:', gold['lay'])
        #     print('')

        if is_tree_eq(self.tgt, gold['tgt'], not_layout=True):
            self.correct['tgt'] = 1

        # if self.correct['lay'] == 1 and self.correct['tgt']==0:
        #     print(' '.join(gold['src']))
        #     print('pred_lay:', self.lay)
        #     print('gold_lay:', gold['lay'])
        #     print('pred_tgt:', self.tgt)
        #     print('gold_tgt:', gold['tgt'])
        #     print('')

        if self.token_prune is not None:
            for tk in self.token_prune:
                if tk in gold['lay']:
                    self.incorrect_prune.add(tk)
