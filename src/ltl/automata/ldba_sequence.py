from ltl.logic import FrozenAssignment

AssignmentSet = frozenset[FrozenAssignment]


class LDBASequence:
    EPSILON = -42  # an arbitrary value that we can put in a tensor and will hopefully never be an actual action

    def __init__(self, reach_avoid: list[tuple[AssignmentSet | type(EPSILON), AssignmentSet]], repeat_last: int = 0):
        """
        :param reach_avoid: A list of pairs of reach and avoid assignments.
        :param repeat_last: Number of times to how often the last reach-avoid pair should be repeated.
        """
        self.reach_avoid = tuple(reach_avoid)
        self.repeat_last = repeat_last

    def __hash__(self):
        return hash(self.reach_avoid)

    def __eq__(self, other):
        if not isinstance(other, LDBASequence):
            return False
        return self.reach_avoid == other.reach_avoid

    def __len__(self):
        return len(self.reach_avoid) + self.repeat_last

    def __iter__(self):
        return iter(self.reach_avoid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start >= len(self.reach_avoid):
                if self.repeat_last <= 0:
                    return []
                return [self.reach_avoid[-1]]
            return self.reach_avoid[item]
        if item >= len(self.reach_avoid):
            if self.repeat_last <= 0:
                raise IndexError
            return self.reach_avoid[-1]
        return self.reach_avoid[item]

    def __repr__(self):
        return f'{self.reach_avoid.__repr__()} x {self.repeat_last}'