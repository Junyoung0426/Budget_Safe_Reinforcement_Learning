import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt



class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)



class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, memory_size=10000, bet_amount=100, initial_money=1000):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.bet_amount = bet_amount
        self.initial_money = initial_money
        self.balance = initial_money
        self.balance_history = [initial_money]
        self.memory = deque(maxlen=memory_size)
        self.win_rate = []

        state_size = len(env.observation_space.sample())
        action_size = env.action_space.n
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, testing=False):
    
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
    
        q_values = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).cpu().detach().numpy().flatten()
    
        if testing:
            max_actions = np.flatnonzero(q_values == q_values.max())
            return np.random.choice(max_actions)
    
        if np.random.rand() < self.epsilon:  
            return self.env.action_space.sample()
        else:  
            max_actions = np.flatnonzero(q_values == q_values.max())
            return np.random.choice(max_actions)
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1, keepdim=True)[0].detach()
        target_q = rewards + self.discount_factor * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=10000, moving_avg_window=500):
        rewards_per_episode = []
        total_wins = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            win = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                episode_reward += reward
                if done and reward > 0:
                    win = True

            self.balance += episode_reward * self.bet_amount
            self.balance_history.append(self.balance)
            rewards_per_episode.append(episode_reward)
            total_wins += int(win)
            win_rate = total_wins / (episode + 1)
            self.win_rate.append(win_rate)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

            if (episode + 1) % 100 == 0:
                print_episode_summary(episode, episode_reward, win_rate, self.balance, self.epsilon)

        moving_avg_rewards = np.convolve(rewards_per_episode, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        plot_results(self.win_rate, moving_avg_rewards, self.balance_history)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.eval()


def plot_results(win_rates, moving_avg_rewards, balance_history):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(range(len(win_rates)), win_rates, label="Win Rate", color="blue")
    axs[0].set_title("Win Rate Over Episodes")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Win Rate")
    axs[0].legend()

    axs[1].plot(range(len(moving_avg_rewards)), moving_avg_rewards, label="Moving Average Reward (500 Episodes)", color="orange")
    axs[1].set_title("Moving Average Reward Over Episodes")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Reward")
    axs[1].legend()

    axs[2].plot(range(len(balance_history)), balance_history, label="Balance", color="green")
    axs[2].set_title("Balance Over Episodes")
    axs[2].set_xlabel("Episodes")
    axs[2].set_ylabel("Balance")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def print_episode_summary(episode, total_reward, win_rate, balance, epsilon=None):
    epsilon_info = f", Epsilon: {epsilon:.5f}" if epsilon is not None else ""
    print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Win Rate: {win_rate:.4f}, Balance: {balance:.2f}{epsilon_info}")
