from pathlib import Path
import gym
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v1')

agent = DQNAgent(env, learning_rate=0.01, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995, batch_size=64, memory_size=10000, bet_amount=100, initial_money=10000000)

balance_history = agent.train(num_episodes=5000)

compare_dir = Path(__file__).resolve().parent.parent.parent  / "compare" / "action_file"
compare_dir.mkdir(exist_ok=True) 

dqn_table_path = compare_dir / "dqn_model.pth"
agent.save_model(dqn_table_path)
print("DQN model saved as dqn_model.pth")
