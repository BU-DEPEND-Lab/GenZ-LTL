# Adapted from Shimmy (MIT License): https://github.com/Farama-Foundation/Shimmy/blob/main/shimmy/dm_control_compatibility.py
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from envs.dmc.dmc_gym_wrapper.wrapper_utils import dm_spec2gym_space, dm_env_step2gym_step
from ltl.logic import FrozenAssignment


class DMCGymWrapper(gym.Env):
    """
    A wrapper from (custom) DeepMind Control Suite environments to the gymnasium API.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, env, render_mode=None):
        self._env = env

        self.observation_space = dm_spec2gym_space(env.observation_spec())
        self.action_space = dm_spec2gym_space(env.action_spec())

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.height = 480
        self.width = 640

    @property
    def dt(self):
        """Returns the environment control timestep which is equivalent to the number of actions per second."""
        return self._env.control_timestep()

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the dm-control environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed=seed)

        timestep = self._env.reset()
        obs, reward, terminated, truncated, info = dm_env_step2gym_step(timestep)
        return obs, info

    def step(
            self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-control environment."""
        timestep = self._env.step(action)
        return dm_env_step2gym_step(timestep)

    def render(self) -> np.ndarray | None:
        """Renders the dm-control env."""
        if self.render_mode == "rgb_array":
            return self._env.physics.render(self.height, self.width)

    def close(self):
        """Closes the environment."""
        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()

    @property
    def np_random(self) -> np.random.RandomState:
        """This should be np.random.Generator but dm-control uses np.random.RandomState."""
        # noinspection PyProtectedMember
        return self._env.task._random

    @np_random.setter
    def np_random(self, value: np.random.RandomState):
        self._env.task._random = value

    def __getattr__(self, item: str):
        """If the attribute is missing, try getting the attribute from dm_control env."""
        return getattr(self._env, item)

    def get_propositions(self) -> list[str]:
        """Returns the propositions of the environment."""
        return self._env.task.get_propositions()

    def get_impossible_assignments(self) -> set[FrozenAssignment]:
        return self._env.task.get_impossible_assignments()
