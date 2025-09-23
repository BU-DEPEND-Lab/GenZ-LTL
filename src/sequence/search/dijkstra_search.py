import math
from dataclasses import dataclass, field
from typing import Optional

from torch import nn

from ltl.automata import LDBA, LDBATransition, LDBASequence
from sequence.search import SequenceSearch
from utils import PriorityQueue


@dataclass(eq=True, frozen=True)
class SearchNode:
    ldba_state: int
    path: LDBASequence
    prev: Optional['SearchNode'] = field(default=None, compare=False)
    prev_transition: Optional[LDBATransition] = field(default=None, compare=False)

    def __lt__(self, other):
        if other is None:
            return True
        return self.ldba_state > other.ldba_state  # arbitrary way to break ties


class DijkstraSearch(SequenceSearch):
    def __init__(self, model: nn.Module, depth: int):
        super().__init__(model)
        self.depth = depth

    def __call__(self, ldba: LDBA, ldba_state: int, obs):
        path = self.dijkstra(ldba, ldba_state, obs)
        path = self.augment_path(ldba, path)
        return path

    def dijkstra(self, ldba: LDBA, ldba_state: int, obs) -> list[LDBATransition]:
        pq = PriorityQueue()
        pq.push(SearchNode(ldba_state, ()), 0)
        visited = set()
        found = False
        goal_node = None
        while not pq.is_empty():
            if found:
                break
            node, cost = pq.pop()
            visited.add(node.ldba_state)
            avoid_transitions = self.collect_avoid_transitions(ldba, node.ldba_state)
            avoid_assignments = frozenset()
            if len(avoid_transitions) > 0:
                avoid_assignments = frozenset(set.union(*[avoid.valid_assignments for avoid in avoid_transitions]))
            for transition in ldba.state_to_transitions[node.ldba_state]:
                if transition.accepting and transition.source == transition.target:
                    found = True
                    goal_node = node
                    break
                if transition in avoid_transitions or transition.target in visited:
                    continue
                next_path = node.path + ((frozenset(transition.valid_assignments), avoid_assignments),)
                if len(next_path) > self.depth:
                    next_path = next_path[-self.depth:]
                new_cost = cost + self.compute_transition_cost(next_path, obs)
                next_node = SearchNode(transition.target, next_path, prev=node, prev_transition=transition)
                if next_node in pq:
                    if new_cost < pq[next_node]:
                        pq.change_priority(next_node, cost)
                else:
                    pq.push(next_node, new_cost)
        assert found and goal_node is not None
        path = []
        while goal_node.prev is not None:
            path.append(goal_node.prev_transition)
            goal_node = goal_node.prev
        path.reverse()
        return path

    def compute_transition_cost(self, path: LDBASequence, obs) -> float:
        if len(path) == 1:
            cost = -math.log(self.get_value(obs, path))
        else:
            cost = math.log(self.get_value(obs, path[:-1])) - math.log(self.get_value(obs, path))
        cost = max(1e-2, cost)
        return cost

    def get_value(self, obs, path) -> float:
        value = super().get_value(path, obs)
        return max(1e-8, min(1.0, value))

    def augment_path(self, ldba: LDBA, path: list[LDBATransition]):
        augmented_path = []
        visited = set()
        for t in path:
            state = t.source
            avoid = set()
            visited.add(state)
            for t2 in ldba.state_to_transitions[state]:
                if t2 == t:
                    continue
                if t2.source == t2.target:
                    continue
                scc = ldba.state_to_scc[t2.target]
                if (scc.bottom and not scc.accepting) or self.only_non_accepting_loops(ldba, t2.target, visited):
                    avoid.update(t2.valid_assignments)
                    continue
            augmented_path.append((t.valid_assignments, avoid))
        return augmented_path
