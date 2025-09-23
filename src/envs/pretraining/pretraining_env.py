from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from ltl.logic import FrozenAssignment, Assignment


class PretrainingEnv(gym.Env):
    metadata = {'render_modes': None}

    def __init__(self, propositions: set[str], impossible_assignments: set[FrozenAssignment]):
        self.propositions = tuple(sorted(propositions))
        self.impossible_assignments = impossible_assignments
        self.observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(0,))
        self.index_to_assignment = {
            i: assignment for i, assignment in enumerate(
                [a for a in Assignment.all_possible_assignments(self.propositions)
                 if a.to_frozen() not in impossible_assignments]
            )
        }
        self.action_space = gym.spaces.Discrete(len(self.index_to_assignment))

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        return np.zeros(shape=(0,)), {'propositions': set()}

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        assignment = self.index_to_assignment[action]
        propositions = {p for p, v in assignment.items() if v}
        return np.zeros(shape=(0,)), 0.0, False, False, {'propositions': propositions}

    def get_propositions(self) -> list[str]:
        return list(self.propositions)

    def get_impossible_assignments(self) -> set[FrozenAssignment]:
        return self.impossible_assignments
