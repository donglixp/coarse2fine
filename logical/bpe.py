
PAR_CHILD_PAIR = 0
ORD_PAIR = 1
UNORD_PAIR = 2


class BpePair(object):
    def __init__(self, _in):
        if isinstance(_in, tuple):
            self.p0, self.p1, self.type_pair = _in
        else:
            raise NotImplementedError

    def is_match(self, c0, c1):
        if self.type_pair in (PAR_CHILD_PAIR, ORD_PAIR):
            return (self.p0, self.p1) == (c0, c1)
        elif self.type_pair == UNORD_PAIR:
            return (self.p0, self.p1) == (c0, c1) or (self.p0, self.p1) == (c1, c0)
        else:
            raise NotImplementedError

    def __key(self):
        return (self.p0, self.p1, self.type_pair)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __str__(self):
        if self.type_pair == PAR_CHILD_PAIR:
            return '({}|{})'.format(self.p0, self.p1).replace('(', '{').replace(')', '}')
        elif self.type_pair in (ORD_PAIR, UNORD_PAIR):
            return '{}|{}'.format(self.p0, self.p1)
        else:
            raise NotImplementedError


def recover_bpe(lay):
    return ' '.join(lay).replace('|', ' ').replace('{', '(').replace('}', ' )').split(' ')


def count_pair(t_list):
    c_dict = {}
    for t in t_list:
        for p in t.all_bpe_pairs():
            c_dict[p] = c_dict.get(p, 0) + 1
    return c_dict


def merge_bpe(pair_best, t_list):
    for t in t_list:
        t.apply_bpe(pair_best)


def learn_bpe(t_list, num_merge, min_freq):
    bpe_list = []
    for __ in range(num_merge):
        # get frequent pair
        c_dict = count_pair(t_list)
        pair_best = max(c_dict.items(), key=lambda x: x[1])
        if pair_best[1] < min_freq:
            break
        bpe_list.append(pair_best)
        # merge trees using the learned pair
        merge_bpe(pair_best[0], t_list)
    return bpe_list


class BpeProcessor(object):
    def __init__(self, bpe_list, enable):
        self.bpe_list = bpe_list
        self.enable = enable

    def process(self, t):
        if not self.enable:
            return
        for p, f in self.bpe_list:
            t.apply_bpe(p)
