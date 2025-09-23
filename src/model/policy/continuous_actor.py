import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from model.mixed_distribution import MixedDistribution
from utils import torch_utils


class ContinuousActor(nn.Module):
    def __init__(self,
                 action_dim: int,
                 layers: list[int],
                 activation: nn.Module | dict[str, any],
                 state_dependent_std: bool = False
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        if isinstance(activation, dict):
            self.hidden_act = activation['hidden']
            self.output_act = activation['output']
        else:
            self.hidden_act = self.output_act = activation
        self.enc = torch_utils.make_mlp_layers(layers, self.hidden_act)
        self.mu = torch_utils.make_mlp_layers([layers[-1], action_dim], self.output_act)
        if self.state_dependent_std:
            self.std = torch_utils.make_mlp_layers([layers[-1], action_dim], self.output_act)
            self.softplus = nn.Softplus()
        else:
            self.logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.epsilon_prob = torch_utils.make_mlp_layers([layers[-1], 1], torch.nn.Sigmoid, final_layer_activation=True)

    def forward(self, obs: torch.tensor) -> MixedDistribution:
        hidden = self.enc(obs)
        mu = self.mu(hidden)
        if self.state_dependent_std:
            std = self.softplus(self.std(hidden))
        else:
            std = self.logstd.expand_as(mu).exp()
        std = std + 1e-3
        normal = Normal(mu, std)
        eps_prob = torch.clamp(self.epsilon_prob(hidden), 1e-3, 1 - 1e-3)
        return MixedDistribution(normal, eps_prob.flatten())
