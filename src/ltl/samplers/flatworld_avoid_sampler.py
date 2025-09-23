import copy
import random
from pprint import pprint


class FlatWorldAvoidSampler:

    @classmethod
    def partial(cls, depth: int, ltl2action_format=False):
        return lambda props: cls(props, depth, ltl2action_format)

    def __init__(self, propositions: list[str], depth: int, ltl2action_format=False):
        """
        :param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
        """
        self.propositions = propositions
        self.depth = depth
        self.ltl2action_format = ltl2action_format

    def __call__(self) -> str:
        d = self.depth if isinstance(self.depth, int) else random.randint(*self.depth)
        num_props = 2 * d
        found = False
        while not found:
            props = random.sample(self.propositions, num_props)
            found = True
            chosen_props = []
            for x in props:
                if isinstance(x, str):
                    chosen_props.append(x)
                else:  # tuple
                    chosen_props.append([y for y in x])
            for i in range(0, len(chosen_props), 2):
                if isinstance(chosen_props[i + 1], list) and chosen_props[i] in chosen_props[i + 1]:
                    found = False
                    break
        props2 = copy.deepcopy(props)
        props = [p if isinstance(p, str) else f"({' & '.join(p)})" for p in props]
        formula = ''
        conjunct = f'!{props.pop()} U {props.pop()}'
        for _ in range(d - 1):
            conjunct = f'!{props.pop()} U ({props.pop()} & ({conjunct}))'
        formula += f'({conjunct})'
        if self.ltl2action_format:
            ltl2action_formula = None
            props2 = [p[0] if isinstance(p, str) else (('and', p[0][0], p[1][0]) if len(
                p) == 2 else ('and', ('and', p[0][0], p[1][0]), p[2][0])) for p in props2]
            ltl2action_conjunct = ('until', ('not', props2.pop()), props2.pop())
            for _ in range(d - 1):
                ltl2action_conjunct = ('until', ('not', props2.pop()), ('and', props2.pop(), ltl2action_conjunct))
            if ltl2action_formula is None:
                ltl2action_formula = ltl2action_conjunct
            else:
                ltl2action_formula = ('and', ltl2action_conjunct, ltl2action_formula)
            return formula, ltl2action_formula
        return formula


if __name__ == '__main__':
    props = ['red', 'magenta', 'blue', 'green', 'aqua', 'yellow', 'orange', ('red', 'magenta'), ('blue', 'green'),
             ('green', 'aqua'), ('blue', 'aqua'), ('blue', 'green', 'aqua')]
    depth = (1, 2)
    sampler = FlatWorldAvoidSampler(props, depth, ltl2action_format=True)

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)
