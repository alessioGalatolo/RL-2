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

from problem1.network import Model


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
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


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
        self.q_network = Model(d_in=n_actions*dim_state,
                               hidden_layers=[128],
                               d_out=1)
        self.target_network = deepcopy(self.q_network)

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
        random_action = super().forward()
        input = torch.Tensor(state)
        torch.cat([inp])
        self.q_network(state)
        clever_action = ...
        if random() > self.epsilon:
            return clever_action
        return random_action
