import torch
from torch.distributions import Distribution

from ltl.automata import LDBASequence


class MixedDistribution(Distribution):
    """
    A class to represent a mixed (categorical and continuous) distribution. We only have a single category (whether
    to take an epsilon transition) and possibly multiple continuous actions.
    """
    arg_constraints = {}

    def __init__(self, dist: Distribution, epsilon_prob: torch.tensor):
        super().__init__(dist.batch_shape, dist.event_shape)
        self.dist = dist
        self.epsilon_prob = epsilon_prob
        self.epsilon_mask = None

    def set_epsilon_mask(self, epsilon_mask):
        self.epsilon_mask = epsilon_mask

    def sample(self, **kwargs) -> torch.tensor:
        sample = self.dist.sample(**kwargs)
        mask = self.epsilon_mask & (torch.rand(len(sample)).to(self.epsilon_prob.device) < self.epsilon_prob)
        sample[mask] = LDBASequence.EPSILON
        return sample

    def log_prob(self, value):
        mask = (value == LDBASequence.EPSILON).all(dim=1)
        assert sum(mask) <= sum(self.epsilon_mask), "More epsilon transitions than allowed."
        log_probs = torch.zeros(len(value), device=value.device)
        log_probs[mask] = torch.log(self.epsilon_prob[mask])
        dist_log_probs = self.dist.log_prob(value)
        if ((1 - self.epsilon_prob) < 1e-6).any():
            print("Warning: epsilon prob is close to 1.")
        log_probs[~mask] = dist_log_probs[~mask].sum(dim=1)
        log_probs[(~mask) & self.epsilon_mask] += torch.log(1 - self.epsilon_prob[(~mask) & self.epsilon_mask])
        return log_probs

    def entropy(self):
        entropy = self.dist.entropy()
        entropy[self.epsilon_mask] *= (1 - self.epsilon_prob[self.epsilon_mask]).unsqueeze(1)
        return entropy

    @property
    def mode(self) -> torch.Tensor:
        epsilon_actions = self.epsilon_mask & (self.epsilon_prob > 0.7)
        result = self.dist.mode
        result[epsilon_actions] = LDBASequence.EPSILON
        return result
