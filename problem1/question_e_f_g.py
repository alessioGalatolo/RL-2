import torch
import gym
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def question_e(q_network):
    ...


def question_f(q_network, env):
    from DQN_agent import CleverAgent
    n_actions = env.action_space.n
    dim_state = len(env.observation_space.high)
    agent_smith = CleverAgent(n_actions, dim_state, torch.device('cpu'))
    steps=20
    ys = np.linspace(0,1.5,steps)
    pis = np.linspace(-np.pi, np.pi,steps)
    Ys, PIs = np.meshgrid(ys,pis)
    state = torch.zeros((1,8))
    max_Qs = np.zeros((steps,steps))
    
    for i_y in range(len(ys)):
        for i_pi in range(len(pis)):
            y, pi = ys[i_y], pis[i_pi]
            state[0,1] = y
            state[0,4] = pi
            q_val = torch.max(agent_smith.get_qvals(state, q_network))
            max_Qs[i_y,i_pi] = q_val

    fig = go.Figure(data=[go.Surface(z=max_Qs, x=ys, y=pis)])

    fig.update_layout(title='Max Qs for the restrictions of the states', autosize=False,
                    width=500, height=500,
                #   margin=dict(l=65, r=50, b=65, t=90)
                )
    fig.update_layout(scene=dict(
                    xaxis_title='y',
                    yaxis_title='angle',
                    zaxis_title='Max (Q)'
                ))
    fig.show()

def question_g(q_network, env: gym.Env, episodes=50):
    from DQN_agent import RandomAgent, CleverAgent
    n_actions = env.action_space.n
    dim_state = len(env.observation_space.high)
    random_agent = RandomAgent(n_actions)
    episodic_reward_random = []
    episodic_reward_smith = []
    print('Testing random agent...')
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            action = random_agent.forward(state)
            next_state, reward, done, _ = env.step(int(action.item()))
            state = next_state
            total_reward += reward
        episodic_reward_random.append(total_reward)

    agent_smith = CleverAgent(n_actions, dim_state, torch.device('cpu'))
    print('Testing clever agent...')
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            action = agent_smith.forward(state, q_network, deterministic=True)
            next_state, reward, done, _ = env.step(int(action.item()))
            state = next_state
            total_reward += reward
        episodic_reward_smith.append(total_reward)
    x = [i for i in range(episodes)]
    plt.plot(x, episodic_reward_random, label="Random agent reward")
    plt.plot(x, episodic_reward_smith, label="Clever agent reward")
    plt.legend()
    plt.title("Total episodic reward for different agents")
    plt.show()
    plt.savefig("agents_total_episodic_reward.png")


def main():
    q_network = torch.load('best_hopeful-sweep-30_.pth', map_location=torch.device('cpu'))
    env = gym.make('LunarLander-v2')
    env.reset()
    # question_e(q_network)
    question_f(q_network, env)
    # question_g(q_network, env)


if __name__ == "__main__":
    main()
