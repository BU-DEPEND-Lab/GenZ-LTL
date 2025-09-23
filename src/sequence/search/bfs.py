from dataclasses import dataclass

from torch import nn

from ltl.automata import LDBA, LDBASequence
from sequence.search import SequenceSearch


@dataclass(eq=True, frozen=True)
class SearchNode:
    ldba_state: int
    sequence: LDBASequence
    visited_states: set[int]


class BFS(SequenceSearch):
    def __init__(self, model: nn.Module, propositions: set[str]):
        super().__init__(model, propositions)

    def __call__(self, ldba: LDBA, ldba_state: int, obs) -> LDBASequence:
        seqs = self.bfs(ldba, ldba_state, obs)
        seq = max(seqs, key=lambda s: self.get_value(s, obs))
        seq = self.augment_sequence(ldba, ldba_state, seq)
        return seq

    def bfs(self, ldba: LDBA, ldba_state: int, obs) -> list[LDBASequence]:
        visited: set[int] = set()
        min_length = 0
        queue = [SearchNode(ldba_state, (), set())]
        sequences = []
        while queue:
            node = queue.pop(0)
            visited.add(node.ldba_state)
            avoid_transitions = self.collect_avoid_transitions(ldba, node.ldba_state, node.visited_states)
            avoid = [a.valid_assignments for a in avoid_transitions]
            avoid = set() if not avoid else set.union(*avoid)
            for t in ldba.state_to_transitions[node.ldba_state]:
                if t.target in visited:
                    continue
                if t.target == t.source and not t.accepting:
                    continue
                scc = ldba.state_to_scc[t.target]
                if scc.bottom and not scc.accepting:
                    continue
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
