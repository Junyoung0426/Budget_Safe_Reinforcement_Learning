from pathlib import Path
import gym
import pickle
from ucb_qlearning.ucb_qlearning_agent import UCBQLearningAgent
from q_learning.q_learning_agent import QLearningAgent
from pathlib import Path
import gym
from ddqn.ddqn_agent import DDQNAgent
from dqn.dqn_agent import DQNAgent
import matplotlib.pyplot as plt



env = gym.make('Blackjack-v1')

qagent = QLearningAgent(env, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, initial_balance=1000000, bet_amount=1000)
qagent.train(num_episodes=50000)

ucb_agent =  UCBQLearningAgent(
    env=env, learning_rate=0.01, discount_factor=0.99, eps=1.0, eps_decay=0.9995, beta=1.0, initial_balance=1000000, bet_amount=1000
)
ucb_agent.train(num_episodes=50000)

dqnagent = DQNAgent(env, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, memory_size=10000, bet_amount=1000, initial_money=1000000)
dqnagent.train(num_episodes=50000)

ddqnagent = DDQNAgent(env, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, memory_size=10000, bet_amount=1000, initial_money=1000000)
ddqnagent.train(num_episodes=50000)



def compare_agents_plot(agents_results):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    for agent_name, (win_rates, balance_history) in agents_results.items():
        axs[0].plot(win_rates, label=f"{agent_name} Win Rate")
    axs[0].set_title("Win Rate Comparison")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Win Rate")
    axs[0].legend()

    for agent_name, (win_rates, balance_history) in agents_results.items():
        axs[1].plot(balance_history, label=f"{agent_name} Balance")
    axs[1].set_title("Balance History Comparison")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Balance")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


agents_results = {
    "UCB Q-Learning": (ucb_agent.win_rate, ucb_agent.balance_history),
    "Q-Learning": (qagent.win_rate, qagent.balance_history),
    "DQN": (dqnagent.win_rate, dqnagent.balance_history),
    "DDQN": (ddqnagent.win_rate,  ddqnagent.balance_history),
}

compare_agents_plot(agents_results)
