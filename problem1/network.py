# authors: Alessio Galatolo & Alfred Nilsson

import torch.nn as nn
import torch
import datetime
import os
from glob import glob

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
    
    def save_checkpoint(self, dir = '.', filename = 'neural-network-1', date=None):
        try:
            if date is None:
                now = datetime.datetime.now()
                now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
            else:
                now = date
            filename += '_' + now + '.pth'
            path = os.path.join(dir, filename)
            save_dict = self.state_dict()
            with open(path, 'wb') as f:
                torch.save(save_dict, f)
        except Exception as e:
            print("Error: ", e)
            quit()
    
    def load_from_checkpoint(self, device, dir = '.', filename = 'neural-network-1', date = '*'):
        filename += '_' + date + '.pth'
        if date == '*':
            # look for checkpoints
            path = os.path.join(dir, filename)
            possible_paths = glob(path)
            path = possible_paths[-1]
        else:
            path = os.path.join(dir, filename)

        try:
            with open(path, 'rb') as f:
                state_dict = torch.load(f, map_location=device)

            self.load_state_dict(state_dict)
        except Exception as e:
            print("Error: ", e)
            quit()
