from functools import partial
from typing import Type, Optional

import torch
from torch import nn

printed_device = False


def get_device():
    global printed_device

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
    if not printed_device:
        print(f'Using device: {device_name}')
        printed_device = True
    return device


def make_mlp_layers(
        layer_sizes: list[int],
        activation: Optional[Type[nn.Module]],
        final_layer_activation=True,
        weight_init=torch.nn.init.orthogonal_,
        bias_init=partial(torch.nn.init.constant_, val=0.0)
) -> nn.Module:
    layers = []
    if activation is None and len(layer_sizes) > 2:
        raise ValueError('Activation function should not be None if there are hidden layers')
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if weight_init is not None:
            weight_init(layers[-1].weight)
        # if bias_init is not None:
        #     bias_init(layers[-1].bias)
        if final_layer_activation or i != len(layer_sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def get_number_of_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
