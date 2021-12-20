from collections import deque
from random import sample
import torch


class ReplayBuffer():
    def __init__(self, size):
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.dones = deque(maxlen=size)

    def __len__(self):
        return len(self.states)

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size):
        states = sample(self.states, batch_size)
        actions = sample(self.actions, batch_size)
        rewards = sample(self.rewards, batch_size)
        next_states = sample(self.next_states, batch_size)
        dones = sample(self.dones, batch_size)

        # states = torch.Tensor(states)
        # actions = torch.Tensor(actions)
        # rewards = torch.Tensor(rewards)
        # next_states = torch.Tensor(next_states)
        # dones = torch.Tensor(dones)

        return states, actions, rewards, next_states, dones
