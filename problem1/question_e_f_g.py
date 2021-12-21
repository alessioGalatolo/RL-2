import torch
import gym
import matplotlib.pyplot as plt


def question_e(q_network):
    ...


def question_f(q_network):
    ...


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
    q_network = torch.load('neural-network-1.pth').cpu()
    env = gym.make('LunarLander-v2')
    env.reset()
    # question_e(q_network)
    # question_f(q_network)
    question_g(q_network, env)


if __name__ == "__main__":
    main()
