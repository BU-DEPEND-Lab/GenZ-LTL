import random
from pprint import pprint


class ReachSampler:

    @classmethod
    def partial(cls, depth: int | tuple[int, int], ltl2action_format=False):
        return lambda props: cls(props, depth, ltl2action_format)

    def __init__(self, propositions: list[str], depth: int | tuple[int, int], ltl2action_format=False):
        """
        :param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
        """
        self.propositions = sorted(propositions)
        self.depth = depth
        self.ltl2action_format = ltl2action_format

    def __call__(self) -> str:
        d = self.depth if isinstance(self.depth, int) else random.randint(*self.depth)
        props = [random.choice(self.propositions)]
        while len(props) < d:
            prop = random.choice(self.propositions)
            if props[-1] != prop:
                props.append(prop)
        formula = f'F {props[0]}'
        for p in props[1:]:
            formula = f'F ({p} & {formula})'
        if self.ltl2action_format:
            ltl2action = ('eventually', props[0][0])
            for p in props[1:]:
                ltl2action = ('eventually', ('and', p[0], ltl2action))
            return formula, ltl2action
        return formula


if __name__ == '__main__':
    props = ['blue', 'green', 'red', 'yellow']
    depth = (1, 3)
    sampler = ReachSampler(props, depth, ltl2action_format=True)

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)
