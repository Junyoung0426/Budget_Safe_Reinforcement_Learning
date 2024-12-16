
from agent import *
from simulate import *
from environment import *
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--exp_setting', type=str)
args = parser.parse_args()


# main result for Blakcjack
if args.exp_setting=='eval_blackjack':
    np.random.seed(args.seed)       
    params1 = {'algorithm':'Ours',
                'eps':0.0, 'eps_decay':(1,1), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
                'beta':(1.0, 0.5), # ucb constant
                'learning_rate':lambda x:1.0/max(x,1), # decay learning rate
                'betting_strategy':'lcb', 'threshold':0.0,
                'possible_bet':[1,2,],
                'initial_budget':100,
                'dynamic_betting':True,
                'dynamic_possible_bet':[0, 20],
    }

    params2 = deepcopy(params1)
    params2['algorithm']='Min. Betting'
    params2['betting_strategy'] = 'min'

    # run simulate
    simulate('blackjack', 
                [params1, params2,], 
                num_simul=100, num_episodes=500, eval_every=10000000000, 
                ci_alpha=0.05,
                line_alpha=[1.0,1.0],
                folder_path='./main_figures/eval_blackjack/',
    )










# main result for Tic-Tac-Toe
if args.exp_setting=='eval_tictactoe':
    np.random.seed(args.seed)
    params1 = {'algorithm':'Ours',
                'eps':0.0, 'eps_decay':(1, 1), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
                'beta':(2.0, 2.0), # ucb/lcb constant
                'learning_rate':lambda x:1.0/max(x,1), # decay learning rate
                'betting_strategy':'lcb', 'threshold':-2.0,
                'possible_bet':[1,2,3,4],
                'initial_budget':300,
                'dynamic_betting':False,
                'dynamic_possible_bet':None,
    }

    params3 = deepcopy(params1)
    params3['algorithm']='Min. Betting'
    params3['betting_strategy'] = 'min'

    params4 = deepcopy(params1)
    params4['algorithm']='Max. Betting'
    params4['betting_strategy'] = 'max'

    # run simulate
    simulate('tictactoe', 
            # [params1,], 
                [params1, params3, params4], 
                num_simul=100, num_episodes=1500, eval_every=10000000000, 
                ci_alpha=0.05,
                line_alpha=[1.0,1.0,1.0],
                folder_path='./main_figures/eval_tictactoe/',
    )

