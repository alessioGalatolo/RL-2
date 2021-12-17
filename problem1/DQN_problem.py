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

# Load packages
from collections import deque
from random import sample
import numpy as np
import gym
import torch
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from DQN_agent import RandomAgent, CleverAgent
from network import Model, ConvNet, SimpleConv
from copy import deepcopy
import wandb
import argparse


parser = argparse.ArgumentParser(description='Train a DQN agent in the Lunar Lander Environment')
parser.add_argument('--wandb',
                        dest='WANDB',
                        action='store_true',
                        default=False)
parser.add_argument('--load-ckpt',
                        dest='CKPT_PATH',
                        default=None,
                        help='Name of the checkpoint file')
parser.add_argument('--net',
                        dest='NET',
                        default='fully-connected',
                        help='Network architecture: \'fully-connected\' or \'conv\' or \'simple-conv\'')
args = parser.parse_args()



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


# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
replay_buffer_size = 10000  # set in range of 5000-30000
n_random_experiences = replay_buffer_size
batch_size_train = 64   # set in range 4-128
max_lr = 1e-3           # set in range 1e-3 to 1e-4
min_lr = 1e-4
CLIP_VAL = 1.5          # a value between 0.5 and 2
C_target = int(replay_buffer_size / batch_size_train) # Target update frequency
start_episode = 0
N_episodes = 800                                # set in range 100 to 1000
discount_factor = 0.8                           # Value of the discount factor
n_ep_running_average = 50                       # Running average of 50 episodes
n_actions = env.action_space.n                  # Number of available actions
dim_state = len(env.observation_space.high)     # State dimensionality
eps_max = 0.99
eps_min = 0.05
decay_period = int(0.8 * N_episodes)
decay_method = 'linear'
LR_decay_period=int(0.9 * N_episodes)
checkpoint_interval = int(N_episodes / 20)
architecture = args.NET                         # 'fully-connected' or 'conv' or 'simple-conv'
hidden_layers = [64, 64] if architecture == 'fully-connected' else None

if args.CKPT_PATH is not None:
    start_episode = int(input('Enter starting episode (affects the exploration param eps): '))
    n_random_experiences = 1
    max_lr = 3e-4
    eps_max = 0.1
    LR_decay_period = int(0.9 * (N_episodes - start_episode))

config = dict(
    replay_buffer_size = replay_buffer_size,  # set in range of 5000-30000
    batch_size_train = batch_size_train,  # set in range 4-128
    max_lr = max_lr, # set in range 1e-3 to 1e-4
    min_lr = min_lr,
    CLIP_VAL = CLIP_VAL, # a value between 0.5 and 2
    C_target = C_target, # Target update frequency
    start_episode = start_episode,
    N_episodes = N_episodes,                            # set in range 100 to 1000
    discount_factor = discount_factor,                      # Value of the discount factor
    n_ep_running_average = n_ep_running_average,                    # Running average of 50 episodes
    n_actions = n_actions,               # Number of available actions
    dim_state = dim_state,  # State dimensionality
    eps_max = eps_max,
    eps_min = eps_min,
    decay_period=decay_period,
    decay_method=decay_method,
    LR_decay_period=LR_decay_period,
    hidden_layers = hidden_layers,
    architecture = architecture
)

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode
train_loss_list = []

# Check if GPU is available
if torch.cuda.is_available():
    print('>>> Using GPU.')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Random agent initialization
agent = CleverAgent(n_actions, dim_state, device, eps_max, eps_min, decay_period=decay_period, decay_method=decay_method)
random_agent = RandomAgent(n_actions)

# Training process
replay_buffer = deque(maxlen=replay_buffer_size)

# Initialize Q network
model_config = dict(d_in = n_actions + dim_state,
                               hidden_layers = hidden_layers,
                               d_out = 1)
if architecture == 'fully-connected':
    q_network = Model(**model_config)
elif architecture == 'conv':
    q_network = ConvNet(**model_config)
elif architecture == 'simple-conv':
    q_network = SimpleConv(**model_config)
else:
    print('Invalid network architecture, choose from \'fully-connected\' or \'conv\' or \'simple-conv\'')
    quit()
q_network.to(device)

if __name__ == '__main__':
    # Load pretrained weights
    if args.CKPT_PATH is not None:
        q_network.load_from_checkpoint(device, filename=args.CKPT_PATH)
    
    # Initialize target
    target_network = deepcopy(q_network)    
    target_network.to(device)

    # Initialize WandB
    mode = 'online' if args.WANDB else 'offline'
    wandb.init(project="Lab2", entity="el2805-rl", config=config, mode=mode )
    run_name = wandb.run.name if args.WANDB else 'local'

    # Initialize optimizers
    optim_q = torch.optim.Adam(q_network.parameters(), lr=max_lr)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_q, LR_decay_period, eta_min = min_lr)

    # Track best model
    best_avg_reward = -np.inf

    #----------------------------------------------------------------------------
    #----------------- Fill replay buffer with random experiences ---------------
    print('Filling replay buffer with random experiences')   

    while len(replay_buffer) < n_random_experiences:
        # Reset environment data and initialize variables
        done = False
        state = env.reset()
        while not done:
            # Take a random action
            action = random_agent.forward(state)
            next_state, reward, done, _ = env.step(int(action.item()))
            replay_buffer.append((state, action, reward, next_state, done))
            # Update state for next iteration
            state = next_state

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    # EPISODES = trange(N_episodes, desc='Episode: ', leave=True, initial=start_episode)
    EPISODES = tqdm(range(start_episode, N_episodes, 1), desc='Episode: ', leave=True, initial=start_episode, total=N_episodes)

    #----------------------------------------------------------------------------
    #-------------------------- Training episodes -------------------------------
    debug = False
    target_net_counter = 1
    for i in EPISODES:
        # Reset environment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = agent.forward(state, q_network)
            
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(int(action.item()))
            replay_buffer.append((state, action, reward, next_state, done))
            # Sample N experience from buffer and update q_net weights

            train_loss = 0
            # FIXME: maybe implent batch update instead
            optim_q.zero_grad()
            for experience in sample(replay_buffer, min(batch_size_train, n_random_experiences)):
                state_train, action_train, reward_train, next_state_train, done_train = experience

                with torch.no_grad():
                    target_qvals = agent.get_qvals(next_state_train, target_network)
                target_val = reward_train + discount_factor * torch.max(target_qvals) * (1 - done_train)

                q_val = agent.get_qvals(state_train, q_network, int(action_train.item()))
                train_loss = train_loss + torch.pow(target_val - q_val, 2)
            
            train_loss = train_loss * (1 / batch_size_train)
            train_loss.backward()
            clip_grad_norm_(q_network.parameters(), CLIP_VAL)
            optim_q.step()
            l = train_loss.detach().cpu().numpy()
            train_loss_list.append(l)
            

            if target_net_counter==C_target:
                target_net_counter = 1
                q_net_state_dict = q_network.state_dict()
                target_network.load_state_dict(q_net_state_dict)

            # Update episode reward
            total_episode_reward += reward
            
            
            # Update state for next iteration
            state = next_state
            t += 1
            target_net_counter += 1

        agent.decay_epsilon(i)
        scheduler.step()
        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
        

        # Close environment
        env.close()

        # Save checkpoint
        if i % checkpoint_interval == 0:
            q_network.save_checkpoint(  dir = 'checkpoints', 
                                        filename='ckpt_' + str(i) + '_' + run_name, 
                                        date='')

        reward_running_avg = running_average(episode_reward_list, n_ep_running_average)[-1]

        # Save best model
        if reward_running_avg > best_avg_reward:
            best_avg_reward = reward_running_avg
            q_network.save_checkpoint(  filename='best_' + run_name, 
                                        date='')

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "LR = {:.1e} - Eps = {:.2f} - Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                scheduler.get_last_lr()[0], agent.epsilon,
                i, total_episode_reward, t,
                reward_running_avg,
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        
        wandb.log({ 'loss':l, 'total_episode_reward':total_episode_reward, 
                    'reward_running_avg':reward_running_avg, 'episode': i}, step=i)

    q_network.save_checkpoint(  dir = 'checkpoints', 
                                        filename='ckpt_' + str(i) + '_' + run_name, 
                                        date='')

    # TODO: save plots for report

    # Plot loss
    train_loss_list = np.array(train_loss_list)
    train_loss_list = np.ravel(train_loss_list)
    print(np.size(train_loss_list))
    plt.plot(train_loss_list)
    plt.show(block=True)
    plt.savefig('loss.png')

    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1 , (N_episodes - start_episode) + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, (N_episodes - start_episode) + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, (N_episodes - start_episode)+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, (N_episodes - start_episode)+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()
