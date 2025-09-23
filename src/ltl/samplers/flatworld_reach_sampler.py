import random
from pprint import pprint


class FlatWorldReachSampler:

    @classmethod
    def partial(cls, depth: int | tuple[int, int], ltl2action_format=False):
        return lambda props: cls(props, depth, ltl2action_format)

    def __init__(self, propositions: list[str], depth: int | tuple[int, int], ltl2action_format=False):
        """
        :param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
        """
        self.propositions = propositions
        self.depth = depth
        self.ltl2action_format = ltl2action_format

    def __call__(self) -> str:
        d = self.depth if isinstance(self.depth, int) else random.randint(*self.depth)
        props = [random.choice(self.propositions)]
        while len(props) < d:
            prop = random.choice(self.propositions)
            if props[-1] != prop:
                props.append(prop)
        formula = f'F {self.prop_to_str(props[0], False)}'
        for p in props[1:]:
            formula = f'F ({self.prop_to_str(p, False)} & {formula})'
        if self.ltl2action_format:
            ltl2action = ('eventually', self.prop_to_str(props[0], True))
            for p in props[1:]:
                ltl2action = ('eventually', ('and',self.prop_to_str(p, True), ltl2action))
            return formula, ltl2action
        return formula

    def prop_to_str(self, prop, ltl2action_format=False):
        if isinstance(prop, tuple):
            if ltl2action_format:
                if len(prop) == 2:
                    return ('and', prop[0][0], prop[1][0])
                else:
                    return ('and', ('and', prop[0][0], prop[1][0]), prop[2][0])
            return f'({" & ".join(prop)})'
        if ltl2action_format:
            return prop[0]
        return prop


if __name__ == '__main__':
    props = ['red', 'magenta', 'blue', 'green', 'aqua', 'yellow', 'orange', ('red', 'magenta'), ('blue', 'green'),
             ('green', 'aqua'), ('blue', 'aqua'), ('blue', 'green', 'aqua')]
    depth = (1, 3)
    sampler = FlatWorldReachSampler(props, depth, ltl2action_format=True)

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)
