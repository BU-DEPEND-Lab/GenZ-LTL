import functools
from typing import Any, SupportsFloat, List
from dataclasses import dataclass
import numpy as np
import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType
from gymnasium import spaces

from envs import get_env_attr
from ltl.automata import ltl2ldba, LDBA, LDBASequence
from ltl.logic import Assignment, FrozenAssignment


@dataclass
class CurrentState:
    """
    Holds one possible path that can be taken in the Buchi automaton.
    """
    state: int
    accepting: bool
    num_accepting_visits: int = 0

    def get_successor(self, state, accepting) -> 'CurrentState':
        return CurrentState(
            state,
            accepting,
            num_accepting_visits=self.num_accepting_visits + int(accepting),
        )


class LDBAWrapper(gymnasium.Wrapper):
    """
    Wrapper that keeps track of LTL goal satisfaction using an LDBA, which is added to the observation space.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
        if "PointLtlSafety" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
            })
        elif "LetterSafetyEnv" in env.spec.id:
            obs_dim = env.observation_space['features'].shape[0]
            self.observation_space = spaces.Dict({
                'features': spaces.Box(0, 1, (obs_dim, obs_dim, 1), dtype=np.float32)
            })
        self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]
        self.region_order = env.get_propositions()
        self.terminate_on_acceptance = False
        self.ldba = None
        # Holds a list of potential current states.
        # The first item is considered the actual "current state".
        self.states = []
        # States that are stuck in a bottom non-accepting SCC. No longer updated.
        self.violating_states = []

    @property
    def ldba_state(self):
        """
        The first item is the de facto current state.
        This property returns its state number (index).
        Returns None if there are no possible states (happens when all paths were violating).
        """
        if self.states:
            return self.states[0].state
        else:
            return None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        # Update trajectory
        props = info['propositions']
        
        # Update the possible states
        prev_state_indices = [s.state for s in self.states]
        # Mapping from (state index, accepting) to state
        # If the same state can be reached in multiple ways, use the trajectory with the most accepting visits
        new_states = {}
        for state in self.states:
            for key in self.ldba.get_next_states(state.state, props):
                successor = state.get_successor(*key)
                # Eliminate the possible states that are violating.
                if self.ldba.is_state_violating(successor.state):
                    self.violating_states.append(successor)
                elif key not in new_states or new_states[key].num_accepting_visits < successor.num_accepting_visits:
                    new_states[key] = successor
        self.states = [new_states[k] for k in sorted(new_states.keys())]

        if not self.states:
            # Violating
            terminated = True
            info['violation'] = True
        elif self.terminate_on_acceptance and (accepting_indices := [i for i, state in enumerate(self.states) if state.accepting]):
            terminated = True
            info['success'] = True
            # Update the self.states such that the accepting one is the first
            if (i := accepting_indices[0]) != 0:
                self.states[0], self.states[i] = self.states[i], self.states[0]
        
        if prev_state_indices != [s.state for s in self.states]:
            # Note that states are sorted by index
            info['ldba_state_changed'] = True

        # Use the best possible trajectory for the number of accepting_visits.
        # This field is used only at the end for infinite-horizon tasks.
        # NOTE: Edge case: A violating state has a larger number of accepting visits than a non-violating state (self.states).
        info['num_accepting_visits'] = max([s.num_accepting_visits for s in (self.states if self.states else self.violating_states)])

        self.complete_observation(obs, info)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        self.info= info
        self.ldba = self.construct_ldba(obs['goal'])
        self.terminate_on_acceptance = self.ldba.is_finite_specification()
        self.states = [
            CurrentState(
                state=self.ldba.initial_state,
                # NOTE: LDBA implementation holds accepting info in transitions.
                # As a result, initial state cannot be accepting.
                accepting=False,
            ),
        ]
        self.violating_states = []
        self.complete_observation(obs, info)
        info['ldba_state_changed'] = True
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None):
        obs['ldba'] = self.ldba
        # Is given to the search algorithm by the agent.
        obs['ldba_states'] = [s.state for s in self.states]
        # Marks whether each ldba state is accepting.
        obs['ldba_states_accepting'] = [s.accepting for s in self.states]
        obs['propositions'] = info['propositions']

    def construct_ldba(self, formula: str) -> LDBA:
        propositions = get_env_attr(self.env, 'get_propositions')()
        ldba = ltl2ldba(formula, propositions, simplify_labels=False)
        possible_assignments = get_env_attr(self.env, 'get_possible_assignments')()
        ldba.prune(possible_assignments)
        ldba.complete_sink_state()
        ldba.compute_sccs()
        initial_scc = ldba.state_to_scc[ldba.initial_state]
        if initial_scc.bottom and not initial_scc.accepting:
            raise ValueError(f'The language of the LDBA for {formula} is empty.')
        return ldba

    def pre_process_obs_zones(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        pre-process the observation
        """
        original_obs = self.task.original_obs
        lidar_dim = self.task.lidar_conf.num_bins
        
        reach_zones = [r.to_string()[0]+"_zones_lidar" for r in list(reach)]
        reach_obs = np.vstack([original_obs[color] for color in reach_zones]) # len(reach_zones) x lidar_dim
        reach_obs = np.max(reach_obs, axis=0) # lidar_dim
        
        ### avoid
        avoid_zones = ["_".join(sorted(l)) + "_zones_lidar" for a in list(avoid) if (l := a.to_string())]
        if len(avoid_zones):
            avoid_obs = np.vstack([original_obs[color] for color in avoid_zones]) # len(avoid_zones) x lidar_dim
            avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
        else:
            avoid_obs = np.zeros(lidar_dim)
            
        agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])
        assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
        return np.concatenate([agent_obs, reach_obs, avoid_obs])
    
    def pre_process_obs_letter(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        pre-process the observation
        """
        obs = self.env.original_obs
        new_obs = np.zeros((obs.shape[0], obs.shape[1]), dtype=obs.dtype)
        letter_to_index = {letter: i for i, letter in enumerate(self.region_order)}
        
        reach_indices = [letter_to_index[r.to_string()[0]] for r in list(reach)]
        avoid_indices = [letter_to_index[a.to_string()[0]] for a in list(avoid) if a.to_string()]
        
        reach_mask = np.any(obs[:, :, reach_indices] > 0, axis=2)
        avoid_mask = np.any(obs[:, :, avoid_indices] > 0, axis=2)
        agent_mask = obs[:, :, -1] > 0
        
        new_obs = np.zeros(obs.shape[:2], dtype=np.float32)
        new_obs[avoid_mask] = 0.5
        new_obs[reach_mask] = 1.0
        new_obs[agent_mask] = 0.2
        return new_obs[..., None]

