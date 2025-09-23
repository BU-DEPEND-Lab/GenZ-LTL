import random

class PartiallyOrderedSampler:

    @classmethod
    def partial(cls, depth: int | list[int], num_conjuncts: int | list[int], disjunct_prob=0.25, as_list=False):
        return lambda props: cls(props, depth, num_conjuncts, disjunct_prob, as_list)

    def __init__(self, propositions: list[str], depth: int | list[int], num_conjuncts: int | list[int],
                 disjunct_prob=0.25, as_list=False):
        self.propositions = sorted(propositions)
        self.depth = depth
        self.num_conjuncts = num_conjuncts
        self.disjunct_prob = disjunct_prob
        self.as_list = as_list

    def __call__(self) -> str | list[list[list[str]]]:
        if self.as_list:
            return self.sample_as_list()
        seqs = [self.sample_sequence() for _ in range(self.num_conjuncts)]
        formulas = [self.sequence_to_formula(seq) for seq in seqs]
        formula = ' & '.join(formulas)
        return formula

    def sample_as_list(self) -> list[list[list[str]]]:
        num_conjuncts = random.randint(*self.num_conjuncts) if isinstance(self.num_conjuncts,
                                                                          list) else self.num_conjuncts
        seqs = [self.sample_sequence() for _ in range(num_conjuncts)]
        # print(' & '.join(map(str, seqs)))
        return seqs

    def sample_sequence(self) -> list[list[str]]:
        seq = []
        depth = random.randint(*self.depth) if isinstance(self.depth, list) else self.depth
        for _ in range(depth):
            population = [p for p in self.propositions if len(seq) == 0 or p not in seq[-1]]
            num_sample = 2 if random.random() < self.disjunct_prob else 1
            seq.append(random.sample(population, num_sample))
        return seq

    def sequence_to_formula(self, seq: list[list[str]]) -> str:
        seq.reverse()
        prop_to_str = lambda prop: prop[0] if len(prop) == 1 else f'({prop[0]} | {prop[1]})'
        formula = f'F {prop_to_str(seq[0])}'
        for prop in seq[1:]:
            formula = f'F ({prop_to_str(prop)} & {formula})'
        return formula




if __name__ == '__main__':
    props = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    depth = 4
    num_conjuncts = 2
    sampler = PartiallyOrderedSampler.partial(depth, num_conjuncts, as_list=False)(props)
    for _ in range(5):
        print(sampler())
