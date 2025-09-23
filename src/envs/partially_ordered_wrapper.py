from typing import Any, SupportsFloat, Callable

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from utils import memory


class PartiallyOrderedWrapper(gymnasium.Wrapper):
    """
    Wrapper that samples partially ordered tasks and adds a subset of the possible sequences to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_task: Callable[[], list[list[list[str]]]]):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
        })
        self.sample_task = sample_task
        self.sequences = None
        self.goal = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        success, changed = self.advance_goal(info['propositions'])
        obs = {'features': obs, 'sequences': self.sequences, 'changed': changed}
        if success:
            info['success'] = True
        terminated = terminated or success
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal = self.sample_task()
        self.sequences = self.goal_to_sequences()
        obs = {'features': obs, 'sequences': self.sequences, 'changed': True}
        return obs, info

    def goal_to_sequences(self):
        heads = [seq[0] for seq in self.goal]
        choices = get_choices(heads)
        seqs = all_paths(choices, max_depth=1)
        seqs = [[(p, 'empty') for p in seq] for seq in seqs]
        return seqs

    def advance_goal(self, propositions: list[str]) -> tuple[bool, bool]:
        if len(propositions) == 0:
            return False, False
        to_remove = []
        changed = False
        for seq in self.goal:
            if any(p in propositions for p in seq[0]):
                seq.pop(0)
                changed = True
                if not seq:
                    to_remove.append(seq)
        for seq in to_remove:
            self.goal.remove(seq)
        if changed and len(self.goal) > 0:
            self.sequences = self.goal_to_sequences()
        success = len(self.goal) == 0
        return success, changed


# @memory.cache
def get_choices(heads: list[set[str]]):
    if len(heads) == 0:
        return [set()]
    choices = []
    rec = get_choices(heads[1:])
    for p in heads[0]:
        for rest in rec:
            choices.append({p} | rest)
    return choices


def all_paths(choices: list[set[str]], max_depth: int = 3):
    paths = set()
    for choice in choices:
        for p in all_choice_paths(choice, max_depth=max_depth):
            paths.add(p)
    return paths


def all_choice_paths(choice: set[str], max_depth: int = 2):
    if max_depth == 0 or len(choice) == 0:
        return [()]
    paths = []
    for p in choice:
        rest = choice - {p}
        for path in all_choice_paths(rest, max_depth - 1):
            paths.append((p,) + path)
    return paths

