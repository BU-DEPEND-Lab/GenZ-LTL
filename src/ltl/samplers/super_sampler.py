import random
from pprint import pprint

from ltl import AvoidSampler
from ltl.samplers.reach_sampler import ReachSampler


class SuperSampler:

    @classmethod
    def partial(cls, *sampler_partials: list[callable]):
        return lambda props: cls(props, sampler_partials)

    def __init__(self, propositions: list[str], sampler_partials: list[callable]):
        self.samplers = [sampler_partial(propositions) for sampler_partial in sampler_partials]

    def __call__(self) -> str:
        return random.choice(self.samplers)()


if __name__ == '__main__':
    props = ['a', 'b', 'c', 'd']
    sampler = SuperSampler(props, [ReachSampler.partial(4), AvoidSampler.partial((1, 2), 1)])

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)