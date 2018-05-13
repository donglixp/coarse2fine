import six
import random as rnd

from bpe import BpePair, PAR_CHILD_PAIR, ORD_PAIR, UNORD_PAIR

TOK_WORD = '<?>'
SKP_WORD = '<sk>'
RIG_WORD = '<]>'

SKIP_OP_LIST = ['lambda', 'exists', 'argmin', 'argmax',
                'min', 'max', 'count', 'sum', 'the']


class STree(object):
    def __init__(self, init):
        self.parent = None
        self.children = []
        self.surr_paren = False

        if init is not None:
            if isinstance(init, list):
                self.set_by_token_list(init)
            elif isinstance(init, six.string_types):
                self.set_by_str(init)
            else:
                raise NotImplementedError

    def add_child(self, c):
        if isinstance(c, type(self)):
            c.parent = self
        self.children.append(c)

    def set_by_str(self, s):
        _tk_list = s.replace('(', '( ').replace('  ', ' ').strip().split(' ')
        self._set_by_token_list(_tk_list)

    def set_by_token_list(self, _tk_list):
        self.set_by_str(' '.join(_tk_list))

    # well-tokenized token list
    def _set_by_token_list(self, _tk_list):
        c_left = _tk_list.count('(')
        c_right = _tk_list.count(')')
        if (c_right > c_left) and (c_right - c_left < len(_tk_list)):
            _tk_list = _tk_list[:c_left - c_right]
        if _tk_list[0] == '(' and _tk_list[-1] == ')':
            tk_list = _tk_list[1:-1]
            self.surr_paren = True
        else:
            tk_list = _tk_list
        i = 0
        while i < len(tk_list):
            if tk_list[i] == '(':
                # find the matching ')'
                depth = 0
                right = len(tk_list)
                for j in range(i + 1, len(tk_list)):
                    if tk_list[j] == ')':
                        if depth == 0:
                            right = j
                            break
                        else:
                            depth -= 1
                    elif tk_list[j] == '(':
                        depth += 1
                c = type(self)(tk_list[i + 1:right])
                c.surr_paren = True
                self.add_child(c)
                i = right + 1
            else:
                # skip ``e'' in ``lambda $0 e''
                if (i < 2) or not ((tk_list[i - 2] == 'lambda') and tk_list[i - 1].startswith('$') and len(tk_list[i]) == 1):
                    self.add_child(tk_list[i])
                i += 1

    def to_list(self, shorten=True):
        s_list = []
        for c in self.children:
            if isinstance(c, type(self)):
                s_list.extend(c.to_list(shorten=shorten))
            else:
                s_list.append(c)
        if self.surr_paren:
            assert len(s_list) > 1
            if shorten:
                s_list = ['(' + s_list[0]] + s_list[1:] + [')']
            else:
                s_list = ['('] + s_list + [')']
        return s_list

    def __str__(self):
        return ' '.join(self.to_list(shorten=True))

    def layout(self, dataset, add_skip=False):
        if dataset == 'atis':
            return self.atis_layout(add_skip=add_skip)
        elif dataset == 'geoqueries':
            return self.atis_layout(add_skip=add_skip)
        elif dataset == 'jobs':
            return self.jobs_layout(add_skip=add_skip)
        else:
            raise NotImplementedError

    def atis_layout(self, add_skip=False):
        op = self.children[0]
        is_leaf = not any((isinstance(c, type(self)) for c in self.children))
        if is_leaf:
            if self.surr_paren:
                op += '@{}'.format(len(self.children) - 1)
                if add_skip:
                    return [op] + [SKP_WORD for __ in range(len(self.children) - 1)] + [RIG_WORD]
                else:
                    return [op]
            else:
                assert len(self.children) == 1
                return [TOK_WORD]
        else:
            skip = 1
            if op in SKIP_OP_LIST:
                # skip one variable ($0, ...)
                skip = 2
            elif op in ('and', 'or', 'not', '=', '>', '<'):
                pass
            else:
                # atis: op in ('airline:e', 'departure_time', 'aircraft', 'fare', 'stop', 'the', 'stops', 'has_meal', 'restriction_code', 'airline_name', 'flight_number', 'capacity')
                op += '@{}'.format(len(self.children) - 1)
            lay_list = [op]
            if add_skip:
                for __ in range(skip - 1):
                    lay_list.append(SKP_WORD)
            for i in range(skip, len(self.children)):
                c = self.children[i]
                if isinstance(c, type(self)):
                    lay_list.extend(c.atis_layout(add_skip=add_skip))
                else:
                    lay_list.append(TOK_WORD)
            if self.surr_paren:
                lay_list = ['(' + lay_list[0]] + lay_list[1:] + [')']
            else:
                raise NotImplementedError
            return lay_list

    def geo_layout(self, add_skip=False):
        pass

    def jobs_layout(self, add_skip=False):
        pass

    def norm(self, not_layout=False):
        if len(self.children) > 1:
            for c in self.children:
                if isinstance(c, type(self)):
                    c.norm(not_layout=not_layout)
            # sort child nodes for ``and/or/=''
            op = self.children[0]
            st_sort = None
            if op in ('and', 'or', '=', 'next_to:t',):
                st_sort = 1
            elif op in ('exists', 'min', 'max', 'count', 'sum'):
                if not_layout:
                    # skip $variable
                    st_sort = 2
                else:
                    st_sort = 1
            if (st_sort is not None) and (len(self.children) > st_sort + 1):
                arg_list = list(sorted(self.children[st_sort:], key=str))
                self.children = self.children[:st_sort] + arg_list

            if not_layout:
                # remove duplicate child nodes for ``and/or''
                if op in ('and', 'or'):
                    deduplicate_list = []
                    has_set = set()
                    for c in self.children:
                        if str(c) not in has_set:
                            has_set.add(str(c))
                            deduplicate_list.append(c)
                    self.children = deduplicate_list
        return self

    def permute(self, not_layout=False):
        if len(self.children) > 1:
            for c in self.children:
                if isinstance(c, type(self)):
                    c.permute(not_layout=not_layout)
            # sort child nodes for ``and/or/=''
            op = self.children[0]
            st_sort = None
            if op in ('and', 'or', '=', 'next_to:t',):
                st_sort = 1
            elif op in ('exists', 'min', 'max', 'count', 'sum'):
                if not_layout:
                    # skip $variable
                    st_sort = 2
                else:
                    st_sort = 1
            if (st_sort is not None) and (len(self.children) > st_sort + 1):
                arg_list = list(self.children[st_sort:])
                rnd.shuffle(arg_list)
                self.children = self.children[:st_sort] + arg_list
        return self

    def is_ordered(self):
        op = self.children[0]
        if op in ('and', 'or', '=', 'next_to:t',) or op in ('exists', 'min', 'max', 'count', 'equals', 'sum'):
            return False
        else:
            return True

    def all_bpe_pairs(self):
        pair_list = []
        if len(self.children) <= 1:
            pass
        elif len(self.children) == 2 and isinstance(self.children[1], six.string_types):
            pair_list.append(
                BpePair((self.children[0], self.children[1], PAR_CHILD_PAIR)))
        else:
            c_list = self.children[1:]
            if len(c_list) >= 2:
                if self.is_ordered():
                    for i in range(len(c_list) - 1):
                        if isinstance(c_list[i], six.string_types) and isinstance(c_list[i + 1], six.string_types):
                            pair_list.append(
                                BpePair((c_list[i], c_list[i + 1], ORD_PAIR)))
                else:
                    s_list = [c for c in c_list if isinstance(
                        c, six.string_types)]
                    s_list.sort()
                    for i in range(len(s_list) - 1):
                        for j in range(i + 1, len(s_list)):
                            pair_list.append(
                                BpePair((s_list[i], s_list[j], UNORD_PAIR)))
            for c in self.children:
                if isinstance(c, type(self)):
                    pair_list.extend(c.all_bpe_pairs())
        return pair_list

    def apply_bpe(self, p):
        is_ordered = self.is_ordered()
        if len(self.children) <= 1:
            pass
        elif p.type_pair == PAR_CHILD_PAIR:
            new_list = []
            for c in self.children:
                if isinstance(c, type(self)) and len(c.children) == 2 and isinstance(c.children[1], six.string_types) and p.is_match(c.children[0], c.children[1]):
                    new_list.append(str(p))
                else:
                    new_list.append(c)
            self.children = new_list
            # check self
            c = self
            if isinstance(c, type(self)) and len(c.children) == 2 and isinstance(c.children[1], six.string_types) and p.is_match(c.children[0], c.children[1]):
                self.children = [str(p)]
                self.surr_paren = False
        elif p.type_pair == ORD_PAIR:
            c_list = self.children[1:]
            if len(c_list) >= 2 and is_ordered:
                new_list = [self.children[0]]
                i = 0
                while i < len(c_list):
                    if (i + 1 < len(c_list)) and isinstance(c_list[i], six.string_types) and isinstance(c_list[i + 1], six.string_types) and p.is_match(c_list[i], c_list[i + 1]):
                        new_list.append(str(p))
                        i += 2
                    else:
                        new_list.append(c_list[i])
                        i += 1
                self.children = new_list
        elif p.type_pair == UNORD_PAIR:
            if not is_ordered:
                while True:
                    c_list = self.children[1:]
                    found = None
                    if len(c_list) >= 2:
                        for i in range(len(c_list) - 1):
                            for j in range(i + 1, len(c_list)):
                                if isinstance(c_list[i], six.string_types) and isinstance(c_list[j], six.string_types) and p.is_match(c_list[i], c_list[j]):
                                    found = (i, j)
                                    break
                            if found is not None:
                                break
                    if found is None:
                        break
                    else:
                        # replace
                        new_list = [self.children[0]] + [c_list[i]
                                                         for i in range(found[0])]
                        new_list.append(str(p))
                        new_list.extend([c_list[i] for i in range(
                            found[0] + 1, len(c_list)) if i != found[1]])
                        self.children = new_list

        for c in self.children:
            if isinstance(c, type(self)):
                c.apply_bpe(p)


def norm_tree_var(t):
    tk_list = t.to_list(shorten=False)
    v_list = []
    for tk in tk_list:
        if tk.startswith('$') and (tk not in v_list):
            v_list.append(tk)
    v_dict = {}
    for i, v in enumerate(v_list):
        v_map = '$' + str(i)
        if v != v_map:
            v_dict[v] = v_map
    return STree([v_dict.get(tk, tk) for tk in tk_list])


def is_tree_eq(t1, t2, not_layout=False):
    try:
        if isinstance(t1, STree):
            t1 = STree(str(t1))
        else:
            t1 = STree(t1)
        if isinstance(t2, STree):
            t2 = STree(str(t2))
        else:
            t2 = STree(t2)
        return str(norm_tree_var(t1.norm(not_layout=not_layout))).lower() == str(norm_tree_var(t2.norm(not_layout=not_layout))).lower()
    except Exception:
        return False


if __name__ == '__main__':
    for s in ("(capacity boeing:mf )",
              "( lambda $0 e ( exists $1 ( and ( airline $1 al0 ) ( flight_number $1 fn0 ) ( = ( aircraft_code $1 ) $0 ) ) ) )",
              "(lambda $0 e ( and ( flight $0 ) ( to $0 ap0 ) ( exists $1 ( and ( city $1 ) ( from $0 $1 ) ) ) ) )".split()):
        t = STree(s)
        print(1, t)
        print(2, t.to_list())
        print(3, t.to_list(shorten=False))
        print(4, ' '.join(t.layout('atis', add_skip=False)))
        print(5, ' '.join(t.layout('atis', add_skip=True)))
        assert len(str(t).split(' ')) == len(t.layout('atis', add_skip=True))
        assert str(t) == str(STree(str(t)))
        t_lay = STree(' '.join(t.layout('atis', add_skip=False)))
        assert str(t_lay) == str(STree(str(t_lay)))
        print('')

    s1 = "(lambda $0 e ( and ( flight $0 ) ( to $0 ap0 ) ( exists $1 ( and ( from $0 $1 ) ( city $1 ) ) ) ) )"
    s2 = "(lambda $a e ( and ( to $a ap0 ) ( flight $a ) ( exists $0 ( and ( city $0 ) ( from $a $0 ) ) ) ) )"
    assert is_tree_eq(STree(s1), STree(s2), not_layout=True)
