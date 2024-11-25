import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=DeprecationWarning)


import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import defaultdict

from evaluation import *





class QLearningAgent:
    def __init__(self, env, params, eval_every):
        # parameters
        self.env = env
        self.possible_bet = params['possible_bet'] # possible betting
        self.dynamic_betting = params['dynamic_betting']
        self.dynamic_possible_bet = params['dynamic_possible_bet']
        self.initial_money = params['initial_budget']
        self.algorithm = params['algorithm']
        self.eps = params['eps']
        self.beta = params['beta']
        self.learning_rate = params['learning_rate']
        self.eps_decay = params['eps_decay']
        self.betting_strategy = params['betting_strategy']
        self.betting_agent = BettingAgent(params['betting_strategy'], params['threshold'])
        self.discount_factor = 1.0
        self.eval_every = eval_every


        # arrays
        self.N_counters = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Q_values_lcb = defaultdict(lambda: np.zeros(env.action_space.n))

        # evaluation metrics
        self.score = [] # (estimated) expection of reward
        self.money_over_time = [self.initial_money]
        self.betting_over_time = []






    def choose_action(self, eps, state):
        if np.random.rand() < eps:
            return self.env.action_space.sample()
        else:
            max_action = np.flatnonzero(self.Q_values[state] == self.Q_values[state].max()) # random selection among argmax Q[state]
            action = np.random.choice(max_action)
            return action






    def update_Q_values(self, state, action, reward, next_state, done, beta, episode):
        future_q_value = 0 if done else np.max(self.Q_values[next_state])

        td_target = reward + self.discount_factor * future_q_value
        
        td_error = td_target - self.Q_values[state][action] + beta[0] * np.sqrt(np.log(max(1,episode)) / max(1, self.N_counters[state][action]))
        self.Q_values[state][action] += self.learning_rate(episode) * td_error

    def update_Q_values_lcb(self, state, action, reward, next_state, done, beta, episode):
        future_q_value = 0 if done else np.max(self.Q_values_lcb[next_state])

        td_target = reward + self.discount_factor * future_q_value
        
        td_error = td_target - self.Q_values_lcb[state][action] - beta[1] * np.sqrt(np.log(max(1,episode)) / max(1, self.N_counters[state][action]))
        self.Q_values_lcb[state][action] += self.learning_rate(episode) * td_error






    def train(self, num_episodes):
        '''
        eps: epsilon greedy search in q-learning
        beta: constant for UCB term
        '''
        beta = self.beta
        eps = self.eps
        eps_decay = self.eps_decay

        for episode in range(num_episodes):
            state, _ = self.env.reset()

            # betting after observe the starting state
            # in fact, in real-world casino we bet before the starting state is observed
            # however, in this work, we bet after the starting state opened to see the learning of betting strategy
            betting = self.betting_agent.bet(state, self.Q_values, self.Q_values_lcb, self.possible_bet, self.money_over_time[-1])
            self.betting_over_time.append(betting)

            done = False

            # if episode==800:
            #     print('here')

            while not done:
                
                # epsilon decay
                if episode%eps_decay[0]==-1:
                    eps = eps * eps_decay[1]

                # action selection
                action = self.choose_action(eps, state) # greedy from ucb
                

                # update (s,a)-counter
                self.N_counters[state][action] += 1
                
                # step
                next_state, reward, done, _, _ = self.env.step(action)

                # update Q-value
                self.update_Q_values(state, action, reward, next_state, done, beta, episode) # Q ucb
                self.update_Q_values_lcb(state, action, reward, next_state, done, beta, episode) # Q lcb
                
                # update current state
                state = next_state

                if not done:
                    if self.dynamic_betting is True:
                        betting += self.betting_agent.bet(state, self.Q_values, self.Q_values_lcb, self.dynamic_possible_bet, self.money_over_time[-1])

            current_money = self.money_over_time[-1] + reward*betting
            self.money_over_time.append(current_money)

            # estimating expected reward
            if episode % self.eval_every==0 or episode==num_episodes-1:
                self.score.append(eval(self.env, self.Q_values, 500))

        self.score = np.array(self.score)
        self.money_over_time = np.array(self.money_over_time)




class BettingAgent:
    def __init__(self, betting_strategy, threshold):
        assert betting_strategy in ['min', 'max', 'random', 'lcb']
        self.betting_strategy = betting_strategy
        self.threshold = threshold

    def bet(self, state, Q_ucb, Q_lcb, possible_bet, current_budget):
        if self.betting_strategy =='min':
            return self.min_bet(possible_bet)
        if self.betting_strategy=='max':
            return self.max_bet(possible_bet)
        if self.betting_strategy=='random':
            return self.random_bet(possible_bet)
        if self.betting_strategy=='lcb':
            return self.lcb_bet(state, Q_ucb, Q_lcb, possible_bet, current_budget, self.threshold)

    def min_bet(self, possible_bet):
        return np.min(possible_bet)

    def max_bet(self, possible_bet):
        return np.max(possible_bet)

    def random_bet(self, possible_bet):
        return np.random.choice(possible_bet)

    def lcb_bet(self, state, Q_ucb, Q_lcb, possible_bet, current_budget, threshold):
        reward_lcb = Q_lcb[state][int(np.argmax(Q_ucb[state]))]
        if reward_lcb==0: return min(possible_bet) # min bet at the begining of learning


        # budget depend
        # budget_lcb = current_budget * np.ones_like(possible_bet) + reward_lcb * np.array(possible_bet)
        
        # budget independent
        budget_lcb = reward_lcb * np.array(possible_bet)

        best_bet = min(possible_bet)
        for i in range(len(possible_bet)):
            if budget_lcb[i] > threshold and possible_bet[i] > best_bet:
                best_bet = possible_bet[i]

        return best_bet
