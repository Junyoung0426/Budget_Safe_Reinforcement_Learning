import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt



class QLearningAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999995, initial_balance=1000, bet_amount=100):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.initial_balance = initial_balance
        self.bet_amount = bet_amount
        self.balance = initial_balance
        self.balance_history = [self.balance]
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            max_action = np.flatnonzero(self.Q[state] == self.Q[state].max())
            action = np.random.choice(max_action)
            return action

    def update_q_values(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

    def train(self, num_episodes=20000, moving_avg_window=500):
        rewards, win_rates = [], []
        total_wins = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            win = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.update_q_values(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if terminated and reward > 0:
                    win = True

            rewards.append(total_reward)
            self.balance += total_reward * self.bet_amount
            self.balance_history.append(self.balance)
            total_wins += int(win)
            win_rate = total_wins / (episode + 1)
            win_rates.append(win_rate)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if (episode + 1) % 1000 == 0:
                eval_reward = self.evaluate_policy(100)  
                print(f"Evaluation after {episode + 1} episodes: Avg Reward: {eval_reward:.2f}")

            if (episode + 1) % 100 == 0:
                print_episode_summary(episode, total_reward, win_rate, self.balance, self.epsilon)

        moving_avg_rewards = np.convolve(rewards, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        plot_results(win_rates, moving_avg_rewards, self.balance_history)

    def evaluate_policy(self, num_episodes=100):
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.Q[state]) 
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
        return total_reward / num_episodes

    def save_q_table(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, filename):
        with open(filename, "rb") as f:
            self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n), pickle.load(f))

def plot_results(win_rates, moving_avg_rewards, balance_history):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(win_rates, label="Win Rate", color="blue")
    axs[0].set_title("Win Rate Over Episodes")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Win Rate")
    axs[0].legend()

    axs[1].plot(moving_avg_rewards, label="Moving Average Reward (500 Episodes)", color="orange")
    axs[1].set_title("Moving Average Reward Over Episodes")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Reward")
    axs[1].legend()

    axs[2].plot(balance_history, label="Balance", color="green")
    axs[2].set_title("Balance Over Episodes")
    axs[2].set_xlabel("Episodes")
    axs[2].set_ylabel("Balance")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def print_episode_summary(episode, total_reward, win_rate, balance, epsilon=None):
    epsilon_info = f", Epsilon: {epsilon:.5f}" if epsilon is not None else ""
    print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Win Rate: {win_rate:.4f}, Balance: {balance:.2f}{epsilon_info}")
