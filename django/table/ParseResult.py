import json
from collections import defaultdict

from tree import SCode, is_code_eq


class ParseResult(object):
    def __init__(self, idx, lay, tgt, token_prune):
        self.idx = idx
        self.lay = lay
        self.tgt = tgt
        self.token_prune = token_prune
        self.correct = defaultdict(lambda: 0)
        self.incorrect_prune = set()

    def eval(self, gold):
        if is_code_eq(self.lay, gold['lay'], not_layout=False):
            self.correct['lay'] = 1
        # else:
        #     print(' '.join(gold['src']))
        #     print('pred:', self.lay)
        #     print('gold:', gold['lay'])
        #     print('')

        if is_code_eq(self.tgt, gold['tgt'], not_layout=True):
            self.correct['tgt'] = 1

        # if self.correct['lay'] == 1 and self.correct['tgt'] == 1 and ('NUMBER' in self.lay and 'STRING' in self.lay and 'NAME' in self.lay):
        # if self.correct['lay'] == 1 and self.correct['tgt'] == 0:
        #     print(' '.join(gold['src']))
        #     print('pred_lay:', ' '.join(self.lay))
        #     print('gold_lay:', ' '.join(gold['lay']))
        #     print('pred_tgt:', ' '.join(self.tgt))
        #     print('gold_tgt:', ' '.join(gold['tgt']))
        #     print('')
