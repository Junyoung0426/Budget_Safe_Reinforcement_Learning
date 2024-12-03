from pathlib import Path 
import gym
from q_learning_agent import QLearningAgent

env = gym.make('Blackjack-v1')

agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999995, initial_balance=1000000, bet_amount=100)

agent.train(num_episodes=50000)

compare_dir = Path(__file__).resolve().parent.parent.parent  / "compare" / "action_file"
compare_dir.mkdir(exist_ok=True) 

q_table_path = compare_dir / "q_learning_model.pkl"
agent.save_q_table(q_table_path)

print(f"Q-learning model saved ")
