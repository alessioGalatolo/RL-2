import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, size, dim_state):
        self.current_index = 0
        self.full = False
        self.size = size
        self.states = np.empty((size, dim_state), dtype=np.float16)
        self.actions = np.empty(size, dtype=object)
        self.rewards = np.empty(size, dtype=np.float16)
        self.next_states = np.empty((size, dim_state), dtype=np.float16)
        self.dones = np.empty(size, dtype=object)
        self.random_generator = np.random.default_rng()

    def __len__(self):
        return self.size if self.full else self.current_index

    def append(self, state, action, reward, next_state, done):
        self.states[self.current_index] = state
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.next_states[self.current_index] = next_state
        self.dones[self.current_index] = done

        self.current_index += 1
        if self.current_index == self.size:
            self.full = True
            self.current_index = 0

    def sample(self, size):
        choices = self.random_generator.choice(self.size, size, replace=False)
        states = self.states[choices]
        actions = self.actions[choices]
        rewards = self.rewards[choices]
        next_states = self.next_states[choices]
        dones = self.dones[choices]

        # states = torch.Tensor(states)
        # actions = torch.Tensor(actions)
        # rewards = torch.Tensor(rewards)
        # next_states = torch.Tensor(next_states)
        # dones = torch.Tensor(dones)

        return states, actions, rewards, next_states, dones
