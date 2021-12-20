from collections import deque
from random import sample
import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, size):
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.dones = deque(maxlen=size)
        self.indices = [i for i in range(size)]

    def __len__(self):
        return len(self.states)

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size):
        inds = sample(self.indices, batch_size)
        states = [self.states[i] for i in inds]
        actions = [self.actions[i] for i in inds]
        rewards = [self.rewards [i] for i in inds]
        next_states = [self.next_states[i] for i in inds]
        dones = [self.dones[i] for i in inds]

        # states = torch.Tensor(states)
        # actions = torch.Tensor(actions)
        # rewards = torch.Tensor(rewards)
        # next_states = torch.Tensor(next_states)
        # dones = torch.Tensor(dones)

        return states, actions, rewards, next_states, dones
