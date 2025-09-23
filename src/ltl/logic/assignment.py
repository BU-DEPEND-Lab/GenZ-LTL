import functools
from typing import MutableMapping

from ltl.logic.boolean_parser import parse


class Assignment(MutableMapping):
    """An assignment of truth values to propositions."""

    def __init__(self, *args, **kwargs):
        self.mapping = {}
        self.update(dict(*args, **kwargs))

    @staticmethod
    @functools.cache
    def all_possible_assignments(propositions: tuple[str, ...]) -> list['Assignment']:
        """Returns all possible assignments for a given set of propositions. Guarantees a deterministic order."""
        p = propositions[0]
        rest = propositions[1:]
        if not rest:
            return [Assignment({p: False}), Assignment({p: True})]
        rest_assignments = Assignment.all_possible_assignments(rest)
        result = [Assignment({p: False}, **assignment) for assignment in rest_assignments]
        result += [Assignment({p: True}, **assignment) for assignment in rest_assignments]
        return result

    @staticmethod
    def more_than_one_true_proposition(propositions: set[str]) -> set['FrozenAssignment']:
        return {
            a.to_frozen()
            for a in Assignment.all_possible_assignments(tuple(propositions))
            if len([v for v in a.values() if v]) > 1
        }

    @staticmethod
    def zero_or_one_propositions(propositions: set[str]) -> list['Assignment']:
        assignments = []
        for p in propositions:
            mapping = {p: True} | {q: False for q in propositions if q != p}
            assignments.append(Assignment(mapping))
        assignments.append(Assignment({p: False for p in propositions}))
        return assignments

    @staticmethod
    def single_proposition(p: str, propositions: set[str]) -> 'Assignment':
        return Assignment({p: True} | {q: False for q in propositions if q != p})

    @staticmethod
    def zero_propositions(propositions: set[str]) -> 'Assignment':
        return Assignment({p: False for p in propositions})

    @staticmethod
    def where(*true_propositions: str, propositions: set[str]):
        return Assignment({p: (p in true_propositions) for p in propositions})

    def satisfies(self, label: str) -> bool:
        if label == 't':
            return True
        ast = parse(label)
        return ast.eval(self.mapping)

    def get_true_propositions(self) -> set[str]:
        return {p for p, v in self.mapping.items() if v}

    def to_frozen(self) -> 'FrozenAssignment':
        return FrozenAssignment(self)

    def __str__(self):
        return str(self.mapping)

    def __repr__(self):
        return repr(self.mapping)

    def __setitem__(self, __key, __value):
        self.mapping[__key] = __value

    def __delitem__(self, __key):
        del self.mapping[__key]

    def __getitem__(self, __key):
        return self.mapping[__key]

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        return iter(self.mapping)

    def __or__(self, other):
        return Assignment({**self.mapping, **other.mapping})

    def __eq__(self, other):
        return self.mapping == other.mapping


class FrozenAssignment:
    """An immutable assignment of truth values to propositions. Used for hashing."""

    def __init__(self, assignment: Assignment):
        self.assignment = frozenset(assignment.items())

    def to_label(self) -> str:
        cnf = [f'{"!" if not truth_value else ""}{p}' for p, truth_value in self.assignment]
        return ' & '.join(sorted(cnf, key=lambda x: x[1:] if x[0] == '!' else x))
    
    def to_string(self) -> str:
        return [f'{p}' for p, truth_value in self.assignment if truth_value]

    def get_true_propositions(self) -> frozenset[str]:
        return frozenset(p for p, v in self.assignment if v)

    def __eq__(self, other):
        return self.assignment == other.assignment

    def __hash__(self):
        return hash(self.assignment)

    def __str__(self):
        return str(self.assignment)

    def __repr__(self):
        active = set(t[0] for t in self.assignment if t[1])
        return ' & '.join(active)

    def __iter__(self):
        return iter(self.assignment)
