import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class UCBQLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, eps=1.0, eps_decay=0.99, beta=1.0, initial_balance=1000, bet_amount=100):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps = eps
        self.eps_decay = eps_decay
        self.beta = beta
        self.initial_balance = initial_balance
        self.bet_amount = bet_amount
        self.balance = initial_balance
        self.win_rate = []
        self.balance_history = [self.balance]
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))  
        self.N_counters = defaultdict(lambda: np.zeros(env.action_space.n)) 


    def choose_action(self, state):
        
        if np.random.rand() < self.eps:  
            return self.env.action_space.sample()
        else:
            ucb_values = [
                self.Q[state][a] + self.beta * np.sqrt(np.log(max(1, sum(self.N_counters[state]))) / max(1, self.N_counters[state][a]))
                for a in range(self.env.action_space.n)
            ]
            max_ucb = np.max(ucb_values)
            best_actions = [a for a in range(self.env.action_space.n) if ucb_values[a] == max_ucb]
            return np.random.choice(best_actions)

    def update_q_values(self, state, action, reward, next_state, done):
        future_q_value = 0 if done else np.max(self.Q[next_state])
        td_target = reward + self.discount_factor * future_q_value
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

    def train(self, num_episodes=20000, moving_avg_window=500):
        rewards = []
        total_wins = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            win = False

            while not done:
                action = self.choose_action(state)
                self.N_counters[state][action] += 1
                next_state, reward, done, _, _ = self.env.step(action)

                self.update_q_values(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done and reward > 0:  
                    win = True

            self.balance += total_reward * self.bet_amount
            self.balance_history.append(self.balance)

            rewards.append(total_reward)
            total_wins += int(win)
            self.win_rate.append(total_wins / (episode + 1))

            self.eps = max(self.eps * self.eps_decay, 0.01)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Win Rate: {self.win_rate[-1]:.4f}, Balance: {self.balance:.2f}")

        moving_avg_rewards = np.convolve(rewards, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        self.plot_results(self.win_rate, moving_avg_rewards, self.balance_history)

    def plot_results(self, win_rates, moving_avg_rewards, balance_history):
       
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

    def save_q_table(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, filename):
        with open(filename, "rb") as f:
            self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n), pickle.load(f))
