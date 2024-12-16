import gym
import gym.envs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm


from agent import *
from environment import *

def simulate(env_name, params, num_simul, num_episodes, eval_every, ci_alpha, line_alpha, folder_path, only_plot=False):
    num_exp = len(params); assert num_exp <= 4


    print(f'\nEnvironment:{env_name}\nRun:{num_simul}\nSave:{folder_path}\nNum. Experiment:{num_exp}')


    algorithm = {i:params[i]['algorithm'] for i in range(num_exp)}
    score = {i:[] for i in range(num_exp)}
    money = {i:[] for i in range(num_exp)}
    betting_over_time = []
    line_color = {'lcb':'green', 'max':'blue', 'min':'red', 'random':'purple'}
    
    for exp in range(num_exp):
        print(f'\n---------- Experiment {exp} ----------')
        for _ in tqdm(range(num_simul)):
            if env_name=='blackjack': env = gym.make('Blackjack-v1')
            elif env_name=='tictactoe': env = TicTacToe(board_size=3)
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



    z = 1.96 # confidence interval constant
    x_reward = np.array(list(range(len(mean_score[0])))) * eval_every
    x_money = np.array(list(range(len(mean_money[0]))))
    color_list = ['blue','green','orange','purple']


    if 'seaborn-whitegrid' in plt.style.available:
        plt.style.use('seaborn-whitegrid')
    elif 'seaborn-v0_8-whitegrid' in plt.style.available:
        plt.style.use('seaborn-v0_8-whitegrid')
    else:
        plt.style.use('classic')
 
    plt.figure(figsize=(4.8, 3))
    ymax = -1
    ymin = 1e10
    for exp in range(num_exp):
        ymax = max(ymax, np.max(mean_money[exp]))
        ymin = min(ymin, np.min(mean_money[exp]))
        plt.plot(x_money, mean_money[exp], color=line_color[params[exp]['betting_strategy']], label=algorithm[exp], alpha=line_alpha[exp])
        plt.fill_between(x_money, mean_money[exp]-z*std_money[exp], mean_money[exp]+z*std_money[exp], color=line_color[params[exp]['betting_strategy']], alpha=ci_alpha, linewidth=0.0)

    plt.ylabel('Budget', fontsize=10)
    plt.xlabel('Episode', fontsize=10)


    if only_plot is False:
        plt.xticks([int(num_episodes*0.25),int(num_episodes*0.50), int(num_episodes*0.75), int(num_episodes*1.0)])

        # plot bust-line
        plt.text(num_episodes*0.6 , 0, 'Bust-Line', fontsize=10, color='black')
        plt.plot(x_money, np.zeros_like(x_money), color='black', linewidth=2.0, linestyle='-.')
        
        plt.legend(loc='best', frameon=True)




    # crop
    plt.xlim([0, num_episodes])
    if ymin<0: plt.ylim([ymin*1.3, ymax*1.1])
    else: plt.ylim([ymin*0.8, ymax*1.1])

    plt.tight_layout()
    plt.savefig(f'{folder_path}{env_name}_budget.png')



    # betting history (not average)
    plt.clf()
    x_betting = np.array(list(range(len(betting_over_time))))
    plt.scatter(x_betting, betting_over_time, color='green', label='Conservative Betting (ours)', s=0.5)
    plt.ylabel('Betting', fontsize=10)
    plt.xlabel('Episode', fontsize=10)
    plt.savefig(f'{folder_path}{env_name}_betting.png')



