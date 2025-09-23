from typing import SupportsFloat, Any

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperActType, WrapperObsType


class DictWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            return
        self.obs_key = 'features'
        self.observation_space = spaces.Dict({self.obs_key: env.observation_space})

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        return {self.obs_key: obs}, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return {self.obs_key: obs}, info
