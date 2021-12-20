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
from random import random
import numpy as np
from torch.nn.modules.loss import MSELoss
from network import Model
import torch

class Agent(object):
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
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        # ''' Compute an action uniformly at random across n_actions possible
        #     choices

        #     Returns:
        #         action (int): the random action
        # '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

class CleverAgent(RandomAgent):
    def __init__(self, dim_state:int, n_hidden:list, n_actions: int, device:torch.DeviceObjType, 
                    lr, decay_period, eps_max=0.99, eps_min=0.05, decay_method='exponential'):
        super().__init__(n_actions)
        self.Q_net = Model(dim_state, n_hidden, n_actions).to(device)
        self.T_net = Model(dim_state, n_hidden, n_actions).to(device)
        self.optim_q = torch.optim.Adam(self.Q_net.parameters(), lr=lr)
        self.device = device
        self.epsilon = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.lr = lr
        self.decay_method = decay_method
        self.target_update_freq = 1
        self.discount_factor = 0.95
        self.decay_period = decay_period
        self.loss_func = MSELoss()

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

    
    def forward(self, state, deterministic = False):
        if random() > self.epsilon * (1 - deterministic):
            state = torch.Tensor(state).unsqueeze(0).to(self.device)
            q_vals = self.Q_net(state)
            clever_action = torch.argmax(q_vals, axis=1)
            return clever_action.item()
        random_action = super().forward(state)
        return random_action

    def train_step(self, rand_exp_batch, episode):
        states, actions, rewards, next_states, dones = rand_exp_batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long)
        next_states = torch.tensor(next_states,dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        batch_inds = torch.arange(len(actions), dtype=torch.long)

        self.optim_q.zero_grad()
        with torch.no_grad():
            t_vals = self.T_net(next_states)
            max_t_vals, _ = torch.max(t_vals, axis=1)
            targets = rewards + self.discount_factor * max_t_vals * torch.logical_not(dones)
        
        q_vals = self.Q_net(states)[batch_inds, actions]

        loss = self.loss_func(q_vals, targets)
        loss.backward()
        self.optim_q.step()

        if episode % self.target_update_freq == 0:
            self.T_net.load_state_dict(self.Q_net.state_dict())
        return loss.detach().cpu()