import random
from pprint import pprint


class AvoidMultipleSampler:

    @classmethod
    def partial(cls, depth: int, num_avoid: int):
        return lambda props: cls(props, depth, num_avoid)

    def __init__(self, propositions: list[str], depth: int, num_avoid: int):
        self.propositions = sorted(propositions)
        self.depth = depth
        self.num_avoid = num_avoid

    def __call__(self) -> str:
        d = self.depth if isinstance(self.depth, int) else random.randint(*self.depth)
        n = self.num_avoid if isinstance(self.num_avoid, int) else random.randint(*self.num_avoid)
        num_props = (n + 1) * d
        props = random.sample(self.propositions, num_props)
        formula = ''
        conjunct = f'!({" | ".join([props.pop() for _ in range(n)])}) U {props.pop()}'
        for _ in range(d - 1):
            conjunct = f'!({" | ".join([props.pop() for _ in range(n)])}) U ({props.pop()} & ({conjunct}))'
        formula += f'({conjunct})'
        return formula


if __name__ == '__main__':
    props = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    depth = 2
    num_avoid = 3
    sampler = AvoidMultipleSampler(props, depth, num_avoid)

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)
