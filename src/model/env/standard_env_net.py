from typing import Type, Optional

import torch
from torch import nn

from utils import torch_utils


class StandardEnvNet(nn.Module):
    def __init__(self, obs_dim: int, layer_sizes: list[int], activation: Optional[Type[nn.Module]]):
        super().__init__()
        self.mlp = torch_utils.make_mlp_layers([obs_dim, *layer_sizes], activation=activation)
        self.embedding_size = layer_sizes[-1]

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.mlp(x)
