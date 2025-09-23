from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from ltl.logic import Assignment


class SafetyGymWrapper(gymnasium.Wrapper):
    """
    A wrapper from safety gymnasium LTL environments to the gymnasium API.
    """

    def __init__(self, env: Any, wall_sensor=True):
        super().__init__(env)
        self.render_parameters.camera_name = 'track'
        self.render_parameters.width = 256
        self.render_parameters.height = 256
        self.num_lidar_bins = env.unwrapped.task.lidar_conf.num_bins
        obs_keys = env.observation_space.spaces.keys()
        self.colors = set()
        for key in obs_keys:
            if key.endswith('zones_lidar'):
                color = key.split('_')[0]
                self.colors.add(color)
        self.observation_space = spaces.Dict(env.observation_space)  # copy the observation space
        if wall_sensor:
            self.observation_space['wall_sensor'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
        self.last_dist = None

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        if 'wall_sensor' in info:
            obs['wall_sensor'] = info['wall_sensor']
        info['propositions'] = {c for c in self.colors if info[f'cost_zones_{c}'] > 0}
        if 'cost_ltl_walls' in info:
            terminated = terminated or info['cost_ltl_walls'] > 0
            reward = -1. if info['cost_ltl_walls'] > 0 else 0.
        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info['propositions'] = []
        obs['wall_sensor'] = np.array([0, 0, 0, 0])
        return obs, info

    def get_propositions(self) -> list[str]:
        return sorted(self.colors)

    def get_possible_assignments(self) -> list[Assignment]:
        return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def get_all_possible_assignments(self) -> list[Assignment]:
        return Assignment.all_possible_assignments(tuple(self.get_propositions()))
