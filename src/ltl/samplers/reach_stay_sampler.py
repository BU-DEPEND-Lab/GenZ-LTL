import random
from pprint import pprint


class ReachStaySampler:

    @classmethod
    def partial(cls):
        return lambda props: cls(props)

    def __init__(self, propositions: list[str]):
        self.propositions = sorted(propositions)

    def __call__(self) -> str:
        prop = random.choice(self.propositions)
        return f'F G {prop}'


