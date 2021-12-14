# authors: Alessio Galatolo & Alfred Nilsson

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, d_in, hidden_layers, d_out):
        super(Model, self).__init__()
        previous_dim = d_in
        self.layers = []
        for n_dim in hidden_layers:
            self.layers.append(nn.Linear(previous_dim, n_dim))
            previous_dim = n_dim
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(previous_dim, d_out))
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)
