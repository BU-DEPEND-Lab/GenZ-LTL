import copy
import random
from pprint import pprint


class AvoidSampler:

    @classmethod
    def partial(cls, depth: int, num_conjuncts: int, ltl2action_format=False):
        return lambda props: cls(props, depth, num_conjuncts, ltl2action_format)

    def __init__(self, propositions: list[str], depth: int, num_conjuncts: int, ltl2action_format=False):
        """
        :param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
        """
        self.propositions = sorted(propositions)
        self.depth = depth
        self.num_conjuncts = num_conjuncts
        self.ltl2action_format = ltl2action_format

    def __call__(self) -> str:
        d = self.depth if isinstance(self.depth, int) else random.randint(*self.depth)
        n = self.num_conjuncts if isinstance(self.num_conjuncts, int) else random.randint(*self.num_conjuncts)
        num_props = 2 * d * n
        props = random.sample(self.propositions, num_props)
        props2 = copy.deepcopy(props)
        formula = ''
        for i in range(n):
            conjunct = f'!{props.pop()} U {props.pop()}'
            for _ in range(d - 1):
                conjunct = f'!{props.pop()} U ({props.pop()} & ({conjunct}))'
            formula += f'({conjunct})'
            if i < n - 1:
                formula += ' & '
        if self.ltl2action_format:
            ltl2action_formula = None
            for i in range(n):
                ltl2action_conjunct = ('until', ('not', props2.pop()[0]), props2.pop()[0])
                for _ in range(d - 1):
                    ltl2action_conjunct = ('until', ('not', props2.pop()[0]), ('and', props2.pop()[0], ltl2action_conjunct))
                if ltl2action_formula is None:
                    ltl2action_formula = ltl2action_conjunct
                else:
                    ltl2action_formula = ('and', ltl2action_conjunct, ltl2action_formula)
            return formula, ltl2action_formula
        return formula


if __name__ == '__main__':
    props = ['blue', 'green', 'red', 'yellow']
    depth = (1, 2)
    num_conjuncts = (1, 1)
    sampler = AvoidSampler(props, depth, num_conjuncts, ltl2action_format=True)

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)

