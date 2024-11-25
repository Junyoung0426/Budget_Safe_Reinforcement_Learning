
from agent import *
from simulate import *
from environment import *
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
args = parser.parse_args()
assert args.env in ['blackjack', 'tictactoe']
env_name = args.env

# env_name = 'tictactoe'
# env_name = 'blackjack'

if __name__=='__main__':
    if env_name=='blackjack':

        # parameter setting
        np.random.seed(4)       
        params1 = {'algorithm':'q-learning (lcb)',
                   'eps':0.0, 'eps_decay':(1,1), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
                   'beta':(1.0, 0.001), # ucb constant
                   'learning_rate':lambda epi:(1.0)/max(1,epi), # decay learning rate
                   'betting_strategy':'lcb', 'threshold':0,
                   'possible_bet':[0,1,],
                   'initial_budget':100,
                   'dynamic_betting':True,
                   'dynamic_possible_bet':[0,1,],
        }

        params2 = deepcopy(params1)
        params2['algorithm']='q-learning (random)'
        params2['betting_strategy'] = 'random'

        params3 = deepcopy(params1)
        params3['algorithm']='q-learning (min)'
        params3['betting_strategy'] = 'min'

        params4 = deepcopy(params1)
        params4['algorithm']='q-learning (max)'
        params4['betting_strategy'] = 'max'

        # run simulate
        simulate('blackjack', 
                 [params1, params2,], 
                 num_simul=30, num_episodes=500, eval_every=10000000000, 
                 alpha=0.1,
                 folder_path='./plot/',
        )





    if env_name=='tictactoe':
        # parameter setting    
           
        np.random.seed(4)
        params1 = {'algorithm':'q-learning (lcb)',
                   'eps':0.0, 'eps_decay':(1, 1), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
                   'beta':(2.0, 2.0), # ucb constant
                   'learning_rate':lambda epi:(1.0)/max(1,epi), # decay learning rate
                   'betting_strategy':'lcb', 'threshold':-1.5,
                   'possible_bet':[1,2,3,4],
                   'initial_budget':300,
                   'dynamic_betting':False,
                   'dynamic_possible_bet':None,
        }

        params2 = deepcopy(params1)
        params2['algorithm']='q-learning (random)'
        params2['betting_strategy'] = 'random'

        params3 = deepcopy(params1)
        params3['algorithm']='q-learning (min)'
        params3['betting_strategy'] = 'min'

        params4 = deepcopy(params1)
        params4['algorithm']='q-learning (max)'
        params4['betting_strategy'] = 'max'

        # run simulate
        simulate('tictactoe', 
                # [params1,], 
                 [params1, params2, params3, params4], 
                 num_simul=30, num_episodes=1000, eval_every=10000000000, 
                 alpha=0.1,
                 folder_path='./plot/',
        )