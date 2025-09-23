import torch
from torch.distributions import Distribution, Categorical

from ltl.automata import LDBASequence


class DiscreteEpsilonDistribution(Distribution):
    """
    A class to represent categorical distribution with an extra category for taking an epsilon transition.
    """
    arg_constraints = {}

    def __init__(self, logits: torch.tensor):
        """
        :param dist: Categorical distribution with n+1 categories, where n is the number of actions.
        """
        super().__init__(logits.shape[:-1])
        self.logits = logits
        self.epsilon_category = logits.shape[1] - 1
        self.epsilon_mask = None

    def set_epsilon_mask(self, epsilon_mask):
        self.epsilon_mask = epsilon_mask

    def sample(self, **kwargs) -> torch.tensor:
        dist = self.create_normalized_distribution()
        sample = dist.sample(**kwargs)
        mask = sample == self.epsilon_category
        sample[mask] = LDBASequence.EPSILON
        return sample

    def log_prob(self, value):
        mask = value == LDBASequence.EPSILON
        assert sum(mask) <= sum(self.epsilon_mask), "More epsilon transitions than allowed."
        value[mask] = self.epsilon_category
        dist = self.create_normalized_distribution()
        return dist.log_prob(value)

    def entropy(self):
        dist = self.create_normalized_distribution()
        return dist.entropy()

    @property
    def mode(self) -> torch.Tensor:
        dist = self.create_normalized_distribution()
        result = dist.mode
        result[result == self.epsilon_category] = LDBASequence.EPSILON
        return result

    def create_normalized_distribution(self):
        logits = self.logits.clone()
        logits[~self.epsilon_mask, self.epsilon_category] = float('-inf')
        return Categorical(logits=logits)
