import random
from pprint import pprint
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment
from envs.flatworld import FlatWorld

flatworld = FlatWorld()
props = set(flatworld.get_propositions())
assignments = flatworld.get_possible_assignments()
assignments.remove(Assignment.zero_propositions(flatworld.get_propositions()))
all_assignments = [frozenset([a.to_frozen()]) for a in assignments]

def get_complete_color_assignments(color: str) -> frozenset[FrozenAssignment]:
    color_assignments = []
    for assignment in flatworld.get_possible_assignments():
        if color in assignment.get_true_propositions():
            color_assignments.append(assignment.to_frozen())
    return frozenset(color_assignments)


for color in ['red', 'magenta', 'blue', 'green', 'aqua']:
    all_assignments.append(get_complete_color_assignments(color))


def get_flatworld_all_assignments() -> list[FrozenAssignment]:
    return all_assignments


def flatworld_all_reach_tasks(depth: int) -> Callable:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reachs = [(a, frozenset()) for a in all_assignments]

        def rec(depth: int):
            if depth == 0:
                return []
            if depth == 1:
                return [[r] for r in reachs]
            rec_res = rec(depth - 1)
            result = []
            for task in rec_res:
                next_reach = task[0][0]
                for p, _ in reachs:
                    if next_reach.issubset(p):
                        continue
                    result.append([(p, frozenset())] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def flatworld_sample_reach(depth: int | tuple[int, int]) -> Callable:
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        reach = random.choice(all_assignments)
        task = [(reach, frozenset())]
        for _ in range(d - 1):
            reach = random.choice([a for a in all_assignments if not reach.issubset(a)])
            task.append((reach, frozenset()))
        return LDBASequence(task)

    return wrapper


def flatworld_all_reach_avoid():
    def wrapper(_):
        seqs = []
        for reach in all_assignments:
            available = [a for a in all_assignments if a != reach and not reach.issubset(a)]
            for avoid in available:
                seqs.append(LDBASequence([(reach, avoid)]))
        return seqs
    return wrapper


def flatworld_sample_reach_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
        not_reach_same_as_last: bool = False
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach):
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
            available = [a for a in all_assignments if a not in last_reach] if not_reach_same_as_last else all_assignments
            reach = random.sample(available, nr)
            available = [a for a in all_assignments if a not in reach and a not in last_reach]
            available = [a for a in available if
                         not any([r.issubset(a) for r in reach])
                         and (len(last_reach) == 0 or not any(
                             [r.issubset(a) for r in last_reach]))]
            if len(available) < na:
                if isinstance(num_avoid, tuple):
                    na = random.randint(num_avoid[0], len(available)) if num_avoid[0] < len(available) else len(
                        available)
                else:
                    raise ValueError('Not enough propositions to sample from')
            avoid = random.sample(available, na)
            reach_assignments = frozenset.union(*reach)
            avoid_assignments = frozenset.union(*avoid) if len(avoid) > 0 else frozenset()
            return reach_assignments, avoid_assignments, reach

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = []
        seq = []
        for _ in range(d):
            reach, avoid, reach_props = sample_one(last_reach)
            seq.append((reach, avoid))
            last_reach = reach_props
        return LDBASequence(seq)

    return wrapper


def flatworld_sample_reach_stay(num_stay: int, num_avoid: tuple[int, int]) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        reach = random.choice(all_assignments)
        # while len(p.get_true_propositions()) > 1:
        #     p = random.choice(assignments)
        na = random.randint(*num_avoid)
        available = [a for a in all_assignments if a != reach and not reach.issubset(a)]
        avoid = random.sample(available, na)
        avoid = frozenset.union(*avoid) if len(avoid) > 0 else frozenset()
        second_avoid = frozenset.union(*all_assignments).difference(reach).union({Assignment.zero_propositions(propositions).to_frozen()})
        task = [(LDBASequence.EPSILON, avoid), (reach, second_avoid)]
        return LDBASequence(task, repeat_last=num_stay)

    return wrapper


if __name__ == '__main__':
    print(flatworld_all_reach_avoid()([]))
