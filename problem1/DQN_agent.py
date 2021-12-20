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
from random import random
import torch
from torch.nn import MSELoss
from networks import Model
from replay_buffer import ReplayBuffer


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
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class AgentSmith(RandomAgent):
    def __init__(self, n_actions: int, dim_state,
                 hidden_layers, device, buffer_size,
                 p_random_action_min, p_random_action_max,
                 discount_factor, n_episodes,
                 learning_rate=1e-4, **kwargs):
        super().__init__(n_actions)
        self.p_random_action_min = p_random_action_min
        self.p_random_action_max = p_random_action_max
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.n_episodes = n_episodes
        self.device = device
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.q_network = Model(dim_state, hidden_layers, n_actions).to(device)
        self.target_network = Model(dim_state, hidden_layers, n_actions).to(device)
        self.update_target()
        self.optim = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def store_experience(self, state, action, reward, new_state, done):
        self.replay_buffer.append(state, action, reward, new_state, done)
        return len(self.replay_buffer) == self.buffer_size

    def p_random_action(self, iteration):
        epsilon = self.p_random_action_max - self.p_random_action_min
        epsilon *= iteration
        epsilon /= self.n_episodes * 0.9
        epsilon = self.p_random_action_max - epsilon
        return min(self.p_random_action_min, epsilon)

    def forward(self, state: np.ndarray, t):
        if random() < self.p_random_action(t):
            return super().forward(state)
        with torch.no_grad():
            state_tensor = torch.Tensor(np.array([state])).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1).indices.item()

    def train(self, batch_size):
        experience = self.replay_buffer.sample(batch_size)
        states, actions, rewards, new_states, dones = experience

        # move to torch
        states = torch.Tensor(np.array(states)).float().to(self.device)
        actions = torch.Tensor(actions).long().to(self.device)
        rewards = torch.Tensor(rewards).float().to(self.device)
        new_states = torch.Tensor(np.array(new_states)).to(self.device)
        dones = torch.Tensor(dones).float().to(self.device)

        self.optim.zero_grad()
        with torch.no_grad():
            new_qs = self.target_network(new_states)
            best_qs = torch.max(new_qs, dim=1).values
            target_qs = rewards + self.discount_factor * best_qs * (1 - dones)
        actual_qs = self.q_network(states)
        qs_actions = actual_qs.gather(1, actions.unsqueeze(1)).squeeze()
        loss = MSELoss()(qs_actions, target_qs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.3)
        self.optim.step()
        return loss.detach().cpu().numpy()

    def save_model(self, path="network.pth"):
        torch.save(self.q_network, path)
