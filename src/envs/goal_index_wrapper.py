from typing import Any, SupportsFloat

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from envs import get_env_attr


class GoalIndexWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a goal index to the observation space. Supported goals are reachability formulae of the form 'F p',
    where p is a proposition in the environment.
    """

    def __init__(self, env: gymnasium.Env, punish_termination=False):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('Goal index wrapper requires dict observations')
        if 'goal' not in env.observation_space.spaces:
            raise ValueError('Goal index wrapper requires goal in observation space')
        propositions = get_env_attr(env, 'get_propositions')()
        self.observation_space['goal_index'] = spaces.Discrete(len(propositions) + 1)
        self.proposition_to_index = {p: i for i, p in enumerate(propositions)}
        self.punish_termination = punish_termination

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        if not self.is_valid(obs['goal']):
            raise ValueError('Fixed embeddings only support reachability goals')

        obs['goal_index'] = self.proposition_to_index[self.goal_to_proposition(obs['goal'])]
        reward = 0.
        if self.punish_termination and terminated:
            reward = -1.
        elif self.goal_to_proposition(obs['goal']) in info['propositions']:
            reward = 1.
            terminated = True
            obs['goal_index'] = len(self.proposition_to_index)  # Indicate that the goal has been reached
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        obs['goal_index'] = self.proposition_to_index[self.goal_to_proposition(obs['goal'])]
        return obs, info

    def is_valid(self, goal: str) -> bool:
        return goal[0] == 'F' and self.goal_to_proposition(goal) in self.proposition_to_index

    @staticmethod
    def goal_to_proposition(goal: str) -> str:
        return goal[1:].strip()
