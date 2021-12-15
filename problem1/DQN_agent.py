# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
from copy import deepcopy
from random import random

from network import Model


class Agent():
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super().__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = torch.Tensor(np.random.randint(0, self.n_actions, size=(1,)))
        return self.last_action
    
    def decay_epsilon(self, iteration):
        pass


class CleverAgent(RandomAgent):
    def __init__(self, n_actions: int, dim_state, eps_max=0.99, eps_min=0.05,
                 decay_period=1, decay_method='exponential'):
        super().__init__(n_actions)
        self.epsilon = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.decay_period = decay_period
        self.decay_method = decay_method
        self.n_actions = n_actions
        self.dim_state = dim_state
        self.q_network = Model(d_in = n_actions + dim_state,
                               hidden_layers = [128],
                               d_out = 1)
        self.target_network = deepcopy(self.q_network)
        self.actions_tensor = torch.eye(n=n_actions, m=n_actions)

    def decay_epsilon(self, iteration):
        new_epsilon = 0
        if self.decay_method == 'exponential':
            new_epsilon = (self.eps_min / self.eps_max) ** ((iteration - 1)
                                                            / (self.decay_period - 1))
            new_epsilon *= self.eps_max
        elif self.decay_method == 'linear':
            new_epsilon = self.eps_max - ((iteration - 1)
                                          * (self.eps_max - self.eps_min))\
                                              / (self.decay_period - 1)
        else:
            print(f'Decay method {self.decay_method} not recognized')
        self.epsilon = max(self.eps_min, new_epsilon)

    def forward(self, state):
        state = torch.Tensor(state).view(1,-1).expand(self.n_actions, self.dim_state)
        random_action = super().forward(state)
        # Each column is a vector [onehot_action, s_1, ..., s_8]
        state_action_tensor = torch.cat((self.actions_tensor, state), dim=1)
        q_vals = self.q_network(state_action_tensor)
        clever_action = torch.argmax(q_vals)
        if random() > self.epsilon:
            return clever_action
        return random_action
