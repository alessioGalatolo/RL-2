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
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
from collections import deque
from random import sample
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import tqdm
from DQN_agent import RandomAgent, CleverAgent
from network import Model
from copy import deepcopy

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
replay_buffer_size = 5000  # set in range of 5000-30000
training_batch = 4  # set in range 4-128
learning_rate = 1e-3 # set in range 1e-3 to 1e-4
N_episodes = 100                             # set in range 100 to 1000
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
eps_max = 0.99
eps_min = 0.05

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = CleverAgent(n_actions, dim_state, eps_max, eps_min, decay_period=int(0.9*N_episodes))
random_agent = RandomAgent(n_actions)

# Training process
replay_buffer = deque(maxlen=replay_buffer_size)

#Initialize networks
q_network = Model(d_in = n_actions + dim_state,
                               hidden_layers = [128],
                               d_out = 1)
target_network = deepcopy(q_network)

#----------------------------------------------------------------------------
#----------------- Fill replay buffer with random experiences ---------------
print('Filling replay buffer with random experiences')

while len(replay_buffer) < replay_buffer_size:
    # Reset environment data and initialize variables
    done = False
    state = env.reset()
    while not done:
        # Take a random action
        action = random_agent.forward(state)
        next_state, reward, done, _ = env.step(int(action.item()))
        replay_buffer.append((state, action, reward, next_state))
        # Update state for next iteration
        state = next_state

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

#----------------------------------------------------------------------------
#-------------------------- Training episodes -------------------------------
debug = True
for i in EPISODES:
    # Reset environment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        # Take a random action
        action = agent.forward(state, q_network)

        if debug and t%8 == 0:
            env.render()
            breakpoint=True
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(int(action.item()))
        replay_buffer.append((state, action, reward, next_state))
        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1

    agent.decay_epsilon(i)
    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
