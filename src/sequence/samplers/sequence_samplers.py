import random
from pprint import pprint
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment


def sample_reach_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
        not_reach_same_as_last: bool = False
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach: set[str]):
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
            available = [p for p in propositions if p not in last_reach] if not_reach_same_as_last else propositions
            reach = random.sample(available, nr)
            available = [p for p in propositions if p not in reach and p not in last_reach]
            if len(available) < na:
                if isinstance(num_avoid, tuple):
                    na = random.randint(num_avoid[0], len(available)) if num_avoid[0] < len(available) else len(
                        available)
                else:
                    raise ValueError('Not enough propositions to sample from')
            avoid = random.sample(available, na)
            reach_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in reach])
            avoid_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in avoid])
            return reach_assignments, avoid_assignments, reach

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = set()
        seq = []
        for _ in range(d):
            reach, avoid, reach_props = sample_one(last_reach)
            seq.append((reach, avoid))
            last_reach = reach_props
        return LDBASequence(seq)

    return wrapper


def all_reach_avoid_tasks(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reach_avoids = [(frozenset([Assignment.single_proposition(p, propositions).to_frozen()]),
                         frozenset([Assignment.single_proposition(q, propositions).to_frozen()]))
                        for p in propositions for q in propositions if p != q]

        def rec(depth: int):
            if depth == 0:
                return []
            if depth == 1:
                return [[ra] for ra in reach_avoids]
            rec_res = rec(depth - 1)
            result = []
            for task in rec_res:
                next_reach, next_avoid = task[0]
                for p, q in reach_avoids:
                    if p == next_reach or p == next_avoid:
                        continue
                    result.append([(p, q)] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def all_reach_tasks(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reachs = [(frozenset([Assignment.single_proposition(p, propositions).to_frozen()]),
                   frozenset())
                  for p in propositions]

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
                    if p == next_reach:
                        continue
                    result.append([(p, frozenset())] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def all_reach_stay_tasks(num_stay: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        tasks = []
        for p in propositions:
            reach = frozenset([Assignment.single_proposition(p, propositions).to_frozen()])
            avoid = frozenset([
                Assignment.zero_propositions(propositions).to_frozen(),
                *[Assignment.single_proposition(q, propositions).to_frozen() for q in propositions if q != p]
            ])
            task = [(LDBASequence.EPSILON, frozenset()), (reach, avoid)]
            tasks.append(LDBASequence(task, repeat_last=num_stay))
        return tasks

    return wrapper


def sample_reach_stay(num_stay: int, num_avoid: tuple[int, int]) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        p = random.choice(propositions)
        reach = frozenset([Assignment.single_proposition(p, propositions).to_frozen()])
        na = random.randint(*num_avoid)
        available = [q for q in propositions if q != p]
        avoid = random.sample(available, na)
        avoid = frozenset([Assignment.single_proposition(q, propositions).to_frozen() for q in avoid])
        second_avoid = frozenset([
            Assignment.zero_propositions(propositions).to_frozen(),
            *[Assignment.single_proposition(q, propositions).to_frozen() for q in propositions if q != p]
        ])
        task = [(LDBASequence.EPSILON, avoid), (reach, second_avoid)]
        return LDBASequence(task, repeat_last=num_stay)

    return wrapper


def fixed(sequence: LDBASequence) -> Callable[[list[str]], Callable[[], LDBASequence]]:
    def wrapper(propositions: list[str]) -> Callable[[], LDBASequence]:
        return lambda: sequence

    return wrapper


if __name__ == '__main__':
    print(sample_reach_stay(100, (0, 2))(['a', 'b', 'c']))
