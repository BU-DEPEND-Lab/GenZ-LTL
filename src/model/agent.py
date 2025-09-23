import numpy as np
import torch

import preprocessing
from model.model import Model
from sequence.search import SequenceSearch
from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment


class Agent:
    def __init__(self, env, model: Model, search: SequenceSearch, propositions: set[str], verbose=False, timeout=None):
        self.env = env
        self.model = model
        self.search = search
        self.propositions = propositions
        self.verbose = verbose
        self.sequence = None
        # Timeout mechanism
        self.timeout = 300 # timeout if timeout else float('inf')
        self.current_goal_steps = 0

    def reset(self):
        self.sequence = None

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:
        if 'ldba_state_changed' in info or self.sequence is None:
            prev_seq = self.sequence
            self.sequence = self.search(obs['ldba'], obs['ldba_states'], obs)
            if self.sequence != prev_seq:
                self.current_goal_steps = 0
            if self.verbose:
                print(f'Selected sequence: {self.sequence}')
        else:
            self.current_goal_steps += 1
            if self.current_goal_steps >= self.timeout:
                # Mark the current subgoal as unfeasible, but not for the accepting states.
                # Because staying in accepting states is the goal; we should never timeout in accepting states.
                unfeasible_states = [s for s, accepting in zip(obs['ldba_states'], obs['ldba_states_accepting']) if not accepting]
                if unfeasible_states:
                    true_props = set()
                    for a in self.sequence[0][0]:
                        true_props = true_props.union(a.get_true_propositions())
                    reach_assignment = Assignment.where(*true_props, propositions=obs['ldba'].propositions).to_frozen()
                    obs['ldba'].mark_unfeasible(unfeasible_states, reach_assignment)
                    prev_seq = self.sequence
                    self.sequence = self.search(obs['ldba'], obs['ldba_states'], obs)
                    # If there are no sequences, an exception will be raised by search.
                    # We don't need to check that condition here.
                    assert self.sequence != prev_seq
                # Reset the step counter for the next timeout.
                self.current_goal_steps = 0
        assert self.sequence is not None
        obs['goal'] = self.sequence
        return self.forward(obs, deterministic)

    def forward(self, obs, deterministic=False) -> np.ndarray:
        
        if self.sequence is not None:
            reach, avoid = self.sequence[0]
            if len(obs["features"].shape) == 1:
                obs["features"] = self.env.pre_process_obs_zones(reach, avoid)
            else:
                obs["features"] = self.env.pre_process_obs_letter(reach, avoid)
        
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        with torch.no_grad():
            dist, _, _ = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()
