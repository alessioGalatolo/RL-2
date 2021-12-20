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
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import AgentSmith
import wandb


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


def main():
    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    state = env.reset()

    # Parameters
    config = dict(
        buffer_size=5000,
        experience_batch_size=32,
        training_frequency=50,  # n of it between training
        n_episodes=500,    # Number of episodes
        p_random_action_min=0.1,
        p_random_action_max=0.99,
        discount_factor=0.85,                     # Value of the discount factor
        learning_rate=1e-3,
        hidden_layer1=64,
        hidden_layer2=64,
    )
    wandb.init(config=config)
    config = wandb.config

    config['hidden_layers'] = [config.hidden_layer1, config.hidden_layer2]
    n_ep_running_average = 50                   # Running average of 50 episodes
    config['n_actions'] = env.action_space.n               # Number of available actions
    config['dim_state'] = len(env.observation_space.high)  # State dimensionality
    experience_batch_size = config['experience_batch_size']
    target_update_frequency = int(config.buffer_size / experience_batch_size)
    config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    N_episodes = config['n_episodes']

    training_frequency = config['training_frequency']  # n of it between training
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    losses = []
    best_avg_reward = -float('Inf')

    # Random agent initialization
    agent = AgentSmith(**config)

    # Fill the buffer
    buffer_full = False
    while not buffer_full:
        action = agent.forward(state, 0)
        next_state, reward, done, _ = env.step(action)
        buffer_full = agent.store_experience(state, action, reward,
                                             next_state, done)
        if done:
            env.close()
            state = env.reset()

    # Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset environment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state, i)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

            if t % training_frequency == 0:
                loss = agent.train(experience_batch_size)
                losses.append(loss)

            if t % target_update_frequency == 0:
                agent.update_target()

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        current_running_average = running_average(episode_reward_list, n_ep_running_average)[-1]
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                current_running_average,
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        if current_running_average != 0 and current_running_average > best_avg_reward:
            best_avg_reward = current_running_average

        if current_running_average > 50:
            break

    agent.save_model()

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
    wandb.log({'best_avg_reward': best_avg_reward})
    print(best_avg_reward)


if __name__ == "__main__":
    main()
