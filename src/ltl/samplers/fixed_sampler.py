import random
from pprint import pprint


class FixedSampler:

    @classmethod
    def partial(cls, formula: str):
        return lambda props: cls(props, formula)

    def __init__(self, props: list[str], formula: str):
        self.formula = formula

    def __call__(self) -> str:
        return self.formula
