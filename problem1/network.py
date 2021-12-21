# authors: Alessio Galatolo & Alfred Nilsson

import torch.nn as nn
import torch
import datetime
import os
from glob import glob


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def save(self, dir=None, filename='neural-network-1', date=None):
        # Saves the object instead of the dict
        try:
            if date is None:
                now = datetime.datetime.now()
                now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
            else:
                now = date
            filename += '_' + now + '.pth'
            if dir is not None:
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path = os.path.join(dir, filename)
            else:
                path = filename
            with open(path, 'wb') as f:
                torch.save(self, f)
        except Exception as e:
            print("Error when saving the whole network: ", e)
            quit()

    def save_checkpoint(self, dir=None, filename='neural-network-1', date=None):
        try:
            if date is None:
                now = datetime.datetime.now()
                now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
            else:
                now = date
            filename += '_' + now + '.pth'
            if dir is not None:
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path = os.path.join(dir, filename)
            else:
                path = filename
            save_dict = self.state_dict()
            with open(path, 'wb') as f:
                torch.save(save_dict, f)
        except Exception as e:
            print("Error when saving the network dict: ", e)
            quit()

    def load_from_checkpoint(self, device, dir='', filename='neural-network-1', date = '*'):
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


class Model(BaseModel):
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

        print('Fully-connected Net with '
              + str(self.get_n_params())
              + ' total params loaded.')

    def forward(self, x):
        return self.network(x)
