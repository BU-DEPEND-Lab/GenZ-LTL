from typing import Callable

from ltl.automata import LDBASequence
from sequence.samplers.curriculum import Curriculum


class CurriculumSampler:

    @classmethod
    def partial(cls, curriculum: Curriculum) -> Callable[[list[str]], 'CurriculumSampler']:
        return lambda propositions: cls(curriculum, propositions)

    def __init__(self, curriculum: Curriculum, propositions: list[str]):
        self.curriculum = curriculum
        self.propositions = propositions

    # def __call__(self):
    #     return self.curriculum.sample(self.propositions)

    def __call__(self, current: str = None):
        return self.curriculum.sample(self.propositions, current=current)
