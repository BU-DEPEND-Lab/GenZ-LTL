from typing import Any, SupportsFloat, Callable

import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment


class SequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
        })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == LDBASequence.EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
            reward = 0.
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif reach != LDBASequence.EPSILON and assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        
        cost = 1.0 if reward == -1. else 0.0
        reach, avoid = self.goal_seq[self.num_reached] \
            if self.num_reached < len(self.goal_seq) else self.goal_seq[-1]
        obs = self.complete_observation(obs, info)
        
        self.obs = obs
        self.info = info
        # obs = self.complete_observation(obs, info)
        return obs, (reward, cost), terminated, truncated, info

    def apply_epsilon_action(self):
        assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        obs = self.complete_observation(obs, info)
        self.obs = obs
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }


class SequenceSafetyWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        if "PointLtlSafety" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
            })
        elif "LetterSafetyEnv" in env.spec.id:
            obs_dim = env.observation_space.shape[0]
            self.observation_space = spaces.Dict({
                'features': spaces.Box(0, 1, (obs_dim, obs_dim, 1), dtype=np.float32)
            })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0 # always 0
        self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"] # from safety_gymnasium
        self.region_order = env.get_propositions()
        self.propositions = set(self.region_order)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        reach, avoid = self.goal_seq[self.num_reached]
        
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        
        reward = 0.0; cost = -1.0; terminated = False
        # reach the "avoid" area
        if assignment in avoid:
            cost = 1.0; info['violation'] = True
            terminated = True
        # reach the "reach" area
        elif assignment in reach:
            reward = 1.0; info['success'] = True
            # sample new subgoal
            self.goal_seq = self.sample_sequence(assignment)
            reach, avoid = self.goal_seq[self.num_reached]
        # reach the boundary of the environment
        elif 'cost_ltl_walls' in info and info['cost_ltl_walls'] > 0:
            cost = 1.0; terminated = True
        
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        self.obs = obs
        self.info = info
        # obs = self.complete_observation(obs, info)
        return obs, (reward, cost), terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        reach, avoid = self.goal_seq[self.num_reached]
        
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        return obs, info

    def pre_process_obs(self, reach, avoid):
        if "PointLtlSafety" in self.env.spec.id:
            obs = self.pre_process_obs_zones(reach, avoid)
        elif "LetterSafetyEnv" in self.env.spec.id:
            obs = self.pre_process_obs_letter(reach, avoid)
        return obs

    def pre_process_obs_zones(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        observation reduction
        """
        original_obs = self.task.original_obs
        lidar_dim = self.task.lidar_conf.num_bins
        agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])

        reach_zones = [r.to_string()[0] + "_zones_lidar" for r in list(reach)]
        avoid_zones = [a.to_string()[0] + "_zones_lidar" for a in list(avoid) if a.to_string()]
        
        reach_obs = np.vstack([original_obs[color] for color in reach_zones])
        reach_obs = np.max(reach_obs, axis=0) # lidar_dim
        if len(avoid_zones):
            avoid_obs = np.vstack([original_obs[color] for color in avoid_zones])
            avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
        else:
            avoid_obs = np.zeros(lidar_dim)
            
        assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
        return np.concatenate([agent_obs, reach_obs, avoid_obs])

    def pre_process_obs_letter(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        observation reduction
        """
        obs = self.env.original_obs
        new_obs = np.zeros((obs.shape[0], obs.shape[1]), dtype=obs.dtype)
        letter_to_index = {letter: i for i, letter in enumerate(self.region_order)}
        
        reach_indices = [letter_to_index[r.to_string()[0]] for r in list(reach)]
        avoid_indices = [letter_to_index[a.to_string()[0]] for a in list(avoid)]
        
        reach_mask = np.any(obs[:, :, reach_indices] > 0, axis=2)
        avoid_mask = np.any(obs[:, :, avoid_indices] > 0, axis=2)
        agent_mask = obs[:, :, -1] > 0
        
        new_obs = np.zeros(obs.shape[:2], dtype=np.float32)
         # specific values do not matter as long as they are distinct
        new_obs[avoid_mask] = 0.5
        new_obs[reach_mask] = 1.0
        new_obs[agent_mask] = 0.2
        return new_obs[..., None]
        
    def complete_observation_current(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq,
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }

