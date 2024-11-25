import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm


from agent import *
from environment import *
from blackjack import *

def simulate(env_name, params, num_simul, num_episodes, eval_every, alpha, folder_path):
    num_exp = len(params); assert num_exp <= 4


    print(f'\nEnvironment:{env_name}\tRun:{num_simul}\tSave:{folder_path}')


    algorithm = {i:params[i]['algorithm'] for i in range(num_exp)}
    score = {i:[] for i in range(num_exp)}
    money = {i:[] for i in range(num_exp)}
    
    for exp in range(num_exp):
        print(f'\n---------- Experiment {exp} ----------')
        for _ in tqdm(range(num_simul)):
            # if env_name=='blackjack': env = gym.make('Blackjack-v1')
            if env_name=='blackjack': env=CardCountingBlackjackEnv()
            elif env_name=='tictactoe': env = TicTacToe(board_size=3)
            elif env_name=='21': env = TwentyOneGameEnv()
            agent = QLearningAgent(env, params[exp], eval_every)
            agent.train(num_episodes)
            score[exp].append(agent.score)
            money[exp].append(agent.money_over_time)
            if agent.betting_strategy=='lcb':
                betting_over_time = agent.betting_over_time

    for exp in range(num_exp):
        score[exp] = np.array(score[exp])
        money[exp] = np.array(money[exp])

    mean_score = {exp:score[exp].mean(axis=0) for exp in range(num_exp)}
    std_score = {exp:score[exp].std(axis=0) for exp in range(num_exp)}
    mean_money = {exp:money[exp].mean(axis=0) for exp in range(num_exp)}
    std_money = {exp:money[exp].std(axis=0) for exp in range(num_exp)}



    z = 1.0
    x_reward = np.array(list(range(len(mean_score[0])))) * eval_every
    x_money = np.array(list(range(len(mean_money[0]))))
    color_list = ['blue','green','orange','purple']


    if 'seaborn-whitegrid' in plt.style.available:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-v0_8-whitegrid' in plt.style.available:
        plt.style.use('seaborn-v0_8-whitegrid')
    else:
        plt.style.use('classic')

    # plt.figure(1)
    # for exp in range(num_exp):
    #     plt.plot(x_reward, mean_score[exp], color=color_list[exp], label=algorithm[exp])
    #     plt.fill_between(x_reward, mean_score[exp] - z*std_score[exp], mean_score[exp] + z*std_score[exp], color=color_list[exp], alpha=alpha, linewidth=0.0)
    # plt.ylabel('Est. Expected Reward', fontsize=10)
    # plt.legend()


    # plt.figure(2)    
    plt.figure(figsize=(4.8, 3))
    ymax = -1
    ymin = 1e10
    for exp in range(num_exp):
        ymax = max(ymax, np.max(mean_money[exp]))
        ymin = min(ymin, np.min(mean_money[exp]))
        plt.plot(x_money, mean_money[exp], color=color_list[exp], label=algorithm[exp])
        plt.fill_between(x_money, mean_money[exp]-z*std_money[exp], mean_money[exp]+z*std_money[exp], color=color_list[exp], alpha=alpha, linewidth=0.0)

    plt.ylabel('Budget', fontsize=10)
    plt.xlabel('Episode', fontsize=10)

    plt.xticks([int(num_episodes*0.25),int(num_episodes*0.50), int(num_episodes*0.75), int(num_episodes*1.0)])

    # plot bust-line
    plt.text(num_episodes*0.6 , -200, 'Bust-Line', fontsize=10, color='black')
    plt.plot(x_money, np.zeros_like(x_money), color='black', linewidth=2.0, linestyle='-.')
    
    plt.legend(loc='best', frameon=True)

    plt.xlim([0, num_episodes])
    plt.ylim([ymin*1.3, ymax*1.1])

    # plt.xlim([0, 500])
    # plt.ylim([-100, 500])

    plt.tight_layout()
    plt.savefig(folder_path)



    # betting history (not average)
    plt.clf()
    x_betting = np.array(list(range(len(betting_over_time))))
    plt.scatter(x_betting, betting_over_time, color=color_list[0], label='q-learning (lcb)')
    plt.show()



