from collections import deque
from random import sample


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

    def sample(self, size):
        states = sample(self.states, size)
        actions = sample(self.actions, size)
        rewards = sample(self.rewards, size)
        next_states = sample(self.next_states, size)
        dones = sample(self.dones, size)

        return states, actions, rewards, next_states, dones
