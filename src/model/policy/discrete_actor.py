from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from model.discrete_epsilon_distribution import DiscreteEpsilonDistribution
from utils import torch_utils


class DiscreteActor(nn.Module):
    def __init__(
            self,
            action_dim: int,
            layers: list[int],
            activation: Optional[nn.Module],
    ):
        super().__init__()
        self.action_dim = action_dim
        # action_dim + 1 to account for epsilon transitions
        self.model = torch_utils.make_mlp_layers([*layers, action_dim + 1], activation,
                                                 final_layer_activation=False)

    def forward(self, obs: torch.tensor) -> torch.distributions.Distribution:
        out = self.model(obs)
        return DiscreteEpsilonDistribution(logits=out)
