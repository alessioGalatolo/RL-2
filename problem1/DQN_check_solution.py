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
from tqdm import trange
from matplotlib import animation
import matplotlib.pyplot as plt

# --------- Added by us ------------
from DQN_problem import legacy_model_loading 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
q_network, agent = legacy_model_loading(device)
q_network.load_from_checkpoint(device, dir='checkpoints/genial-grass-35', filename='ckpt_799*')
visualize = True
# visualize = False
save_gif = False


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    plt.figure()

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, 
                                   frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=24)
# ---------- End added by us ------------


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

# # Load model
# try:
#     model = torch.load('neural-network-1.pth')
#     print('Network model: {}'.format(model))
# except:
#     print('File neural-network-1.pth not found!')
#     exit(-1)

# Import and initialize Mountain Car Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50

# Reward
episode_reward_list = []  # Used to store episodes reward

# Anim frames
frames = []

# Simulate episodes
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        # q_values = model(torch.tensor([state]))

        # _, action = torch.max(q_values, axis=1)
        with torch.no_grad():
            action = agent.forward(state, q_network, deterministic=True)
        
        next_state, reward, done, _ = env.step(int(action.item()))

        # Update episode reward
        total_episode_reward += reward

        if visualize:
            frames.append(env.render(mode="rgb_array"))
            if done:
                print(f'{t} steps taken - end reward = {reward} - tot episode reward = {total_episode_reward}')

        # Update state for next iteration
        state = next_state

        t += 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

if avg_reward - confidence >= CONFIDENCE_PASS:
    print('Your policy passed the test!')
else:
    print("Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence".format(CONFIDENCE_PASS))

if save_gif and visualize:
    save_frames_as_gif(frames)
