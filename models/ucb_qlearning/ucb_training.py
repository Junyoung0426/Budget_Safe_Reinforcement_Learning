from pathlib import Path
import gym
import pickle
from ucb_qlearning_agent import UCBQLearningAgent

env = gym.make('Blackjack-v1')

ucb_agent =  UCBQLearningAgent(
    env=env, learning_rate=0.1, discount_factor=0.99, eps=1.0, eps_decay=0.99, beta=1.0, initial_balance=1000, bet_amount=100
)
ucb_agent.train(num_episodes=50000)
compare_dir = Path(__file__).resolve().parent.parent.parent  / "compare" / "action_file"
compare_dir.mkdir(exist_ok=True)  

ucb_qtable_path= compare_dir / "ucb_qlearning_model.pkl"
ucb_agent.save_q_table(ucb_qtable_path)

print(f"UCB Q-learning model saved as {ucb_qtable_path}")
