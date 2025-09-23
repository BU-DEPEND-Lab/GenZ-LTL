import copy
from dataclasses import dataclass

from torch import nn

from ltl.automata import LDBA, LDBASequence
from ltl.logic import Assignment
from sequence.search import SequenceSearch


@dataclass(eq=True, frozen=True)
class SearchNode:
    ldba_state: int
    sequence: LDBASequence
    visited_states: set[int]


class ThresholdBFS(SequenceSearch):
    def __init__(self, model: nn.Module, value_threshold: float = 0.3):
        super().__init__(model)
        self.value_threshold = value_threshold

    def __call__(self, ldba: LDBA, ldba_state: int, obs) -> LDBASequence:
        seqs = self.bfs(ldba, ldba_state)
        updates_seqs = self.compute_threshold_sequences(seqs, ldba, ldba_state, obs)
        seq = max(updates_seqs, key=lambda s: self.get_value(s, obs))
        return seq

    def compute_threshold_sequences(self, seqs: list[LDBASequence], ldba: LDBA, ldba_state: int, obs):
        """
        Computes the updated sequences based on the value threshold.
        """
        updates_seqs = []
        for seq in seqs:
            new_seq = []
            state = ldba_state
            for i in range(len(seq)):
                new_avoid = set()
                value = self.get_value(seq[i:], obs)
                next_state = None
                for transition in ldba.state_to_transitions[state]:
                    if transition.valid_assignments == seq[i][0]:
                        next_state = transition.target
                        continue
                    if transition.source == transition.target:
                        continue
                    alternative = transition.target
                    scc = ldba.state_to_scc[alternative]
                    if scc.bottom and not scc.accepting:
                        alternative_value = -1.0
                    else:
                        alternative_seqs = self.bfs(ldba, alternative)
                        alternative_value = max([self.get_value(seq, obs) for seq in alternative_seqs])
                    if value - alternative_value > self.value_threshold:
                        new_avoid.update(transition.valid_assignments)
                assert next_state is not None
                new_seq.append((seq[i][0], frozenset(new_avoid)))
                state = next_state
            updates_seqs.append(tuple(new_seq))
        return updates_seqs

    def bfs(self, ldba: LDBA, ldba_state: int) -> list[LDBASequence]:
        visited: set[int] = set()
        min_length = 0
        queue = [SearchNode(ldba_state, (), set())]
        sequences = []
        while queue:
            node = queue.pop(0)
            visited.add(node.ldba_state)
            for t in ldba.state_to_transitions[node.ldba_state]:
                if t.target in visited and not t.accepting:
                    continue
                if t.target == t.source and not t.accepting:
                    continue
                scc = ldba.state_to_scc[t.target]
                if scc.bottom and not scc.accepting:
                    continue
                avoid = set()
                for t2 in ldba.state_to_transitions[node.ldba_state]:
                    if t2.source != t2.target and t2 != t:
                        avoid.update(t2.valid_assignments)
                new_sequence = node.sequence + ((frozenset(t.valid_assignments), frozenset(avoid)),)
                if min_length > 0 and len(new_sequence) > min_length:
                    assert all(len(n.sequence) >= min_length for n in queue)
                    break
                if scc.accepting:
                    assert all(len(s) == len(new_sequence) for s in sequences)
                    sequences.append(new_sequence)
                    min_length = len(new_sequence)
                    continue
                new_node = SearchNode(t.target, new_sequence, node.visited_states | {node.ldba_state})
                queue.append(new_node)
        return sequences
