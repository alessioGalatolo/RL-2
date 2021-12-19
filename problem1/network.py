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


class ConvNet(BaseModel):
    def __init__(self, d_in, hidden_layers, d_out):
        super().__init__()
        layers = [nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
                  nn.ReLU(),
                  nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
                  nn.Flatten(),
                  nn.ReLU(),
                  nn.Linear(16, 1)]
        self.network = nn.Sequential(*layers)
        print('ConvNet with '
              + str(self.get_n_params())
              + ' total params loaded.')

    def forward(self, x):
        return self.network(x.unsqueeze(1))


class SimpleConv(BaseModel):
    def __init__(self, d_in, hidden_layers, d_out):
        super().__init__()
        layers = [nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=3),
                  nn.ReLU(),
                  nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3),
                  nn.Flatten(),
                  nn.ReLU(),
                  nn.Linear(32, 1)]
        self.network = nn.Sequential(*layers)
        print('ConvNet with '
              + str(self.get_n_params())
              + ' total params loaded.')

    def forward(self, x):
        return self.network(x.unsqueeze(1))


# # Test code
# import torch.functional as F
# import torch.nn as nn
# a = torch.randn(4,1,12)
# x1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)(a)
# print(x1.shape)
# x2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)(x1)
# print(x2.shape)
# x3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)(x2)
# print(x3.shape)
# x4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)(x3)
# print(x4.shape)
