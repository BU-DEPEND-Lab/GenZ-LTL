from typing import Type

import torch
from torch import nn

from utils import torch_utils


class SetNetwork(nn.Module):

    def __init__(self, input_dim: int, layer_sizes: list[int], activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.mlp = torch_utils.make_mlp_layers([input_dim, *layer_sizes], activation=activation,
                                               weight_init=None)
        self.embedding_size = layer_sizes[-1]

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.sum(dim=-2)
        return self.mlp(x)
