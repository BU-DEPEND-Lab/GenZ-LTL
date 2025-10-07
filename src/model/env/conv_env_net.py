from typing import Type

import torch
from torch import nn


class ConvEnvNet(nn.Module):
    def __init__(
            self,
            obs_dim: tuple[int, int, int],
            channels: list[int],
            kernel_size: tuple[int, int],
            activation: Type[nn.Module]
    ):
        super().__init__()
        h, w, c = obs_dim
        channels = [c, *channels]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size))
            layers.append(activation())
        layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)
        num_conv = len(channels) - 1
        self.embedding_size = (h - num_conv) * (w - num_conv) * channels[-1]

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.einsum("bhwc->bchw", x)
        return self.conv(x)
