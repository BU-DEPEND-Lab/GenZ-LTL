from abc import ABC, abstractmethod

import torch
from torch import nn

import preprocessing
from ltl.automata import LDBA, LDBASequence, LDBATransition


class SequenceSearch(ABC):
    """A search that can be performed on an LDBA and yields the optimal sequence according to the model."""

    def __init__(self, model: nn.Module, propositions, **kwargs):
        self.model = model
        self.propositions = propositions

    @abstractmethod
    def __call__(self, ldba: LDBA, ldba_state: int, obs) -> LDBASequence:
        pass

    def get_value(self, seq: LDBASequence, obs) -> float:
        obs['goal'] = seq
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        _, value, cost_value = self.model(preprocessed)
        return value.item()
    
    # for rco
    def get_value_safety(self, seq: LDBASequence, obs) -> float:
        obs['goal'] = seq
        reach, avoid = seq[0]
        if len(obs["features"].shape) == 1:
            obs["features"] = self.env.pre_process_obs_zones(reach, avoid)
        else:
            obs["features"] = self.env.pre_process_obs_letter(reach, avoid)
        
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        with torch.no_grad():
            _, value, cost_value, lag = self.model(preprocessed, collect=False)
        return value.item() - lag.item() * cost_value.item()

    @staticmethod
    def collect_avoid_transitions(ldba: LDBA, state: int, visited_ldba_states: set[int]) -> set[LDBATransition]:
        avoid = set()
        for transition in ldba.state_to_transitions[state]:
            if transition.source == transition.target:
                continue
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting or transition.target in visited_ldba_states:
                avoid.add(transition)
        return avoid

    def augment_sequence(self, ldba: LDBA, ldba_state: int, seq: LDBASequence) -> LDBASequence:
        """
        Augments the sequence to avoid transitions that lead to non-accepting loops.
        """
        augmented_path = []
        visited = set()
        state = ldba_state
        for reach, a in seq:
            visited.add(state)
            avoid = set()
            found = False
            for t in ldba.state_to_transitions[state]:
                if t.valid_assignments == reach:
                    state = t.target
                    found = True
                    continue
                if t.source == t.target:
                    continue
                scc = ldba.state_to_scc[t.target]
                if (scc.bottom and not scc.accepting) or self.only_non_accepting_loops(ldba, t.target, visited):
                    avoid.update(frozenset(t.valid_assignments))
            assert found
            assert a.issubset(avoid)
            augmented_path.append((reach, frozenset(avoid)))
        return tuple(augmented_path)

    def only_non_accepting_loops(self, ldba: LDBA, state: int, visited: set[int]) -> bool:
        if state in visited:
            return True
        stack = [state]
        marked = set()
        while stack:
            state = stack.pop()
            for t in ldba.state_to_transitions[state]:
                if t.target in marked:
                    continue
                scc = ldba.state_to_scc[t.target]
                if scc.bottom and not scc.accepting:
                    continue
                if t.target in visited:
                    continue
                if t.accepting:
                    return False
                stack.append(t.target)
            marked.add(state)
        visited.update(marked)
        return True
