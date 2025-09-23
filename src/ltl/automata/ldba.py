import functools
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from ltl.logic import FrozenAssignment, Assignment
from utils import to_sympy, simplify, sympy_to_str


class LDBA:
    def __init__(self, propositions: set[str], formula: Optional[str] = None, simplify_labels=True):
        self.formula = formula
        self.propositions: tuple[str, ...] = tuple(sorted(propositions))
        self.simplify_labels = simplify_labels
        self.num_states = 0
        self.num_transitions = 0
        self.initial_state = None
        self.state_to_transitions: dict[int, list[LDBATransition]] = {}
        self.state_to_incoming_transitions: dict[int, list[LDBATransition]] = {}
        self.sink_state: Optional[int] = None
        self.complete = False
        self.possible_assignments: Optional[list[Assignment]] = None
        self.state_to_scc = {}

    @property
    def states(self) -> list[int]:
        return list(range(self.num_states))

    def add_state(self, state: int, initial=False):
        if state < 0:
            raise ValueError('State must be a positive integer.')
        if initial:
            if self.initial_state is not None:
                raise ValueError('Initial state already set.')
            self.initial_state = state
        self.num_states = max(self.num_states, state + 1)
        if state not in self.state_to_transitions:
            self.state_to_transitions[state] = []
        if state not in self.state_to_incoming_transitions:
            self.state_to_incoming_transitions[state] = []

    def get_next_states(self, state: int, propositions: set[str], take_epsilon=False) -> List[Tuple[int, bool]]:
        """
        Given the current state and the propositions that are true,
        returns a list of possible next states and whether each taken transition is accepting.
        """
        # take_epsilon = True
        if take_epsilon:
            eps_transitions = [t for t in self.state_to_transitions[state] if t.is_epsilon()]
            # print(f"eps_transitions: {eps_transitions}")
            if len(eps_transitions) > 1:
                raise ValueError('More than one epsilon transition from a state is currently not supported.')
            assert eps_transitions
            t = eps_transitions[0]
            return [(t.target, t.accepting)]
        assignment = Assignment({p: (p in propositions) for p in self.propositions}).to_frozen()
        outcomes = [
            (transition.target, transition.accepting)
            for transition in self.state_to_transitions[state]
            if assignment in transition.valid_assignments
        ]
        if not outcomes:
            raise ValueError('Invalid transition.')
        return outcomes

    def is_state_violating(self, state: int) -> bool:
        """
        Returns true if the state belongs to a bottom SCC that doesn't have any accepting states.
        """
        scc = self.state_to_scc[state]
        return scc.bottom and not scc.accepting

    def contains_state(self, state: int) -> bool:
        return state <= self.num_states

    def is_finite_specification(self) -> bool:
        """
        Checks if the LDBA represents a finite specification.
        Note: this is not an exhaustive check, but a sufficient heuristic for LDBAs constructed by rabinizer.
        """
        if not self.state_to_scc:
            self.compute_sccs()
        accepting_sccs = [scc for scc in self.state_to_scc.values() if scc.accepting]
        if len(accepting_sccs) > 1:
            return False
        scc = accepting_sccs[0]
        return scc.bottom and len(scc.states) == 1

    def add_transition(self, source: int, target: int, label: Optional[str], accepting: bool) -> 'LDBATransition':
        if source < 0 or source >= self.num_states:
            raise ValueError('Source state must be a valid state index.')
        if target < 0 or target >= self.num_states:
            raise ValueError('Target state must be a valid state index.')
        if self.simplify_labels and label is not None:
            label = sympy_to_str(simplify(to_sympy(label)))
        transition = LDBATransition(source, target, label, accepting, self.propositions)
        self.num_transitions += 1
        for t in self.state_to_transitions[source]:
            if t == transition:
                raise ValueError('There can only be a single transition between two states. Consider merging labels.')
        self.state_to_transitions[source].append(transition)
        self.state_to_incoming_transitions[target].append(transition)
        return transition

    def check_valid(self) -> bool:
        """Checks that the LDBA satisfies the following conditions:
           - It has a deterministic first component
           - It has a deterministic second component
           - All transitions from the first to the second component are epsilon transitions
           - There are no other epsilon transitions
           - All transitions from the second component stay in the second component
           - All accepting transitions are in the second component
           - The first component may be empty
           - The LDBA is fully connected
        """
        if self.initial_state is None:
            return False
        first_visited = set()
        first_queue = [self.initial_state]
        second_states = set()
        found_accepting = False
        while first_queue:
            state = first_queue.pop(0)
            first_visited.add(state)
            if not self.check_deterministic_transitions(state):
                return False
            for transition in self.state_to_transitions[state]:
                if transition.is_epsilon():
                    if transition.target in first_visited:
                        return False  # epsilon transition in the first component
                    second_states.add(transition.target)
                else:
                    if transition.target in second_states:
                        return False  # transition from first to second component is not epsilon
                    if transition.target not in first_visited:
                        first_queue.append(transition.target)
                if transition.accepting:
                    found_accepting = True
        if found_accepting and len(second_states) > 0:
            return False  # accepting transition in the first component
        second_queue = list(second_states)
        second_visited = set()
        while second_queue:
            state = second_queue.pop(0)
            second_visited.add(state)
            if not self.check_deterministic_transitions(state):
                return False
            for transition in self.state_to_transitions[state]:
                if transition.is_epsilon():
                    return False  # epsilon transition in the second component
                if transition.target in first_visited:
                    return False  # transition back from second to first component
                if transition.target not in second_visited:
                    second_queue.append(transition.target)
                if transition.accepting:
                    found_accepting = True
        visited = first_visited | second_visited
        if len(visited) < self.num_states:
            return False  # not fully connected
        return found_accepting

    def check_deterministic_transitions(self, state: int) -> bool:
        """Checks that the transitions from a state are deterministic."""
        num_assignment_transitions = Counter([
            a for transition in self.state_to_transitions[state] for a in transition.valid_assignments
        ])
        return all(c <= 1 for c in num_assignment_transitions.values())

    def complete_sink_state(self):
        """Adds a sink state and transitions to the LDBA if any transitions are missing."""
        if self.complete:
            return
        sink_state = self.num_states
        if self.possible_assignments:
            all_assignments = set([a.to_frozen() for a in self.possible_assignments])
        else:
            all_assignments = set(
                [a.to_frozen() for a in Assignment.all_possible_assignments(tuple(self.propositions))])
        for state in range(self.num_states):
            covered_assignments = set() if not self.state_to_transitions[state] else set.union(
                *[t.valid_assignments for t in self.state_to_transitions[state]]
            )
            if len(covered_assignments) != len(all_assignments):
                # missing transitions - need to add sink state
                if not self.has_sink_state():
                    self.sink_state = sink_state
                    self.add_state(sink_state)
                    t = self.add_transition(sink_state, sink_state, 't', False)
                    assert self.has_sink_state()
                sink_assignments = all_assignments - covered_assignments
                sink_label = self.valid_assignments_to_label(sink_assignments)
                t = self.add_transition(state, sink_state, sink_label, False)
        self.complete = True

    def has_sink_state(self) -> bool:
        return self.sink_state is not None

    def valid_assignments_to_label(self, valid_assignments: set[FrozenAssignment]) -> str:
        assert len(valid_assignments) > 0
        formula = ' | '.join('(' + a.to_label() + ')' for a in valid_assignments)
        if not self.simplify_labels:
            return formula
        simplified = simplify(to_sympy(formula))
        return sympy_to_str(simplified)

    def prune(self, possible_assignments: list[Assignment]):
        """Prunes transitions that involve impossible assignments. Impossible assignments may be derived from knowledge
           of the underlying MDP."""
        self.possible_assignments = possible_assignments
        to_remove = set()
        for transitions in self.state_to_transitions.values():
            for t in transitions:
                if t.is_epsilon():
                    continue
                t.valid_assignments = {a.to_frozen() for a in possible_assignments if a.satisfies(t.label)}
                t.feasible_assignments = t.valid_assignments.copy()
                if t.valid_assignments:
                    t.label = self.valid_assignments_to_label(t.valid_assignments)
                else:
                    to_remove.add(t)
        self.num_transitions -= len(to_remove)
        for state in range(self.num_states):
            self.state_to_transitions[state] = [t for t in self.state_to_transitions[state] if t not in to_remove]
            self.state_to_incoming_transitions[state] = [t for t in self.state_to_incoming_transitions[state]
                                                         if t not in to_remove]

    def compute_sccs(self) -> None:
        """Computes the strongly connected components of the LDBA using Tarjan's algorithm."""
        if self.state_to_scc:
            return
        num = 0
        nums: dict[int, int] = {}
        visited: set[int] = set()
        stack: list[tuple[int, set[int]]] = []
        active: set[int] = set()

        def tarjan(s: int):
            nonlocal num
            nonlocal nums
            nonlocal visited
            nonlocal stack
            nonlocal active
            visited.add(s)
            active.add(s)
            num += 1
            nums[s] = num
            stack.append((s, {s}))
            for t in self.state_to_transitions[s]:
                if t.target not in visited:
                    tarjan(t.target)
                elif t.target in active:
                    scc = set()
                    while True:
                        u, current = stack.pop()
                        scc |= current
                        if nums[u] <= nums[t.target]:
                            break
                    stack.append((u, scc))
            if stack[-1][0] == s:
                _, states = stack.pop()
                active -= states
                transitions = [t for state in states for t in self.state_to_transitions[state]]
                accepting = any(t.accepting and t.target in states for t in transitions)
                bottom = all(t.target in states for t in transitions)
                scc = SCC(frozenset(states), accepting, bottom)
                for state in states:
                    assert state not in self.state_to_scc
                    self.state_to_scc[state] = scc

        tarjan(self.initial_state)
    
    def mark_unfeasible(self, states: list[int], assignment: FrozenAssignment):
        """
        Marks the given assignment unfeasible for the given state list.
        """
        for state in states:
            for t in self.state_to_transitions[state]:
                try:
                    t.feasible_assignments.remove(assignment)
                except KeyError:
                    continue


@dataclass
class LDBATransition:
    source: int
    target: int
    label: Optional[str]  # None for epsilon transitions
    accepting: bool
    propositions: tuple[str, ...]
    valid_assignments: set[FrozenAssignment] = field(init=False)
    feasible_assignments: set[FrozenAssignment] = field(init=False)

    def __post_init__(self):
        if self.label is None:  # epsilon
            self.valid_assignments = set()
            self.feasible_assignments = set()
        else:
            self.valid_assignments = {
                a.to_frozen()
                for a in Assignment.all_possible_assignments(self.propositions)
                if a.satisfies(self.label)
            }
            self.feasible_assignments = self.valid_assignments.copy()

    def is_epsilon(self) -> bool:
        return self.label is None

    def is_feasible(self) -> bool:
        return self.label is None or self.feasible_assignments

    @property
    def positive_label(self) -> str:
        if self.is_epsilon():
            return 'ε'
        return ' | '.join(
            '&'.join(
                k
                for k, v in a.assignment
                if v)
            if sum(v for _, v in a.assignment) > 0
            else '{}'
            for a in self.valid_assignments
        )

    def __eq__(self, other):
        if not isinstance(other, LDBATransition):
            return False
        return (self.source == other.source
                and self.target == other.target
                and self.is_epsilon() == other.is_epsilon()
                and self.accepting == other.accepting)

    def __hash__(self):
        return hash((self.source, self.target, self.is_epsilon(), self.accepting))


@dataclass(eq=True, frozen=True)
class SCC:
    states: frozenset[int]
    accepting: bool
    bottom: bool
