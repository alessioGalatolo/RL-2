import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim) -> None:
        super().__init__()
        last_layer_dim = input_dim
        layers = []
        for layer_dim in hidden_layers:
            layers.append(nn.Linear(last_layer_dim, layer_dim))
            layers.append(nn.ReLU())
            last_layer_dim = layer_dim
        layers.append(nn.Linear(last_layer_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
