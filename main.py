
from agent import *
from simulate import *
from environment import *
import argparse
from copy import deepcopy

# parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str)
# args = parser.parse_args()
# assert args.env in ['blackjack', 'tictactoe']
# env_name = args.env






env_name = '21'

if __name__=='__main__':

    # python3 main.py --env blackjack
    if env_name=='blackjack':
        # seeds
        np.random.seed(4)


        # parameter setting       
        params1 = {'algorithm':'q-learning (lcb)',
                   'eps':0.0, 'eps_decay':(1000,0), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
                   'beta':(1.0, 0.001), # ucb constant
                   'learning_rate':lambda epi:0.1, # decay learning rate
                   'betting_strategy':'lcb', 'threshold':0,
                   'possible_bet':[0,1],
                   'initial_budget':100,
                   'dynamic_betting':True,
                   'dynamic_possible_bet':[0,1],
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
                 [params1, ], 
                 num_simul=1, num_episodes=1000, eval_every=10000000000, 
                 alpha=0.0,
                 folder_path='./plot/blackjack.png',
        )

    if env_name=='21':
        # seeds
        np.random.seed(4)


        # parameter setting       
        params1 = {'algorithm':'q-learning (lcb)',
                   'eps':0.0, 'eps_decay':(1000,0), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
                   'beta':(1.0, 1.0), # ucb constant
                   'learning_rate':lambda epi:0.1, # decay learning rate
                   'betting_strategy':'lcb', 'threshold':0,
                   'possible_bet':[1,2],
                   'initial_budget':100,
                   'dynamic_betting':False,
                   'dynamic_possible_bet':[],
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
        simulate('21', 
                 [params1, params2, params3, params4], 
                 num_simul=20, num_episodes=500, eval_every=10000000000, 
                 alpha=0.0,
                 folder_path='./plot/21.png',
        )


    # python3 main.py --env tictactoe
    if env_name=='tictactoe':
        
        # seeds
        np.random.seed(4)


        # parameter setting       
        params1 = {'algorithm':'q-learning (lcb)',
                   'eps':0.0, 'eps_decay':(500, 0.9), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
                   'beta':(2.0, 2.0), # ucb constant
                   'learning_rate':lambda epi:(1.0)/max(1,epi), # decay learning rate
                   'betting_strategy':'lcb', 'threshold':40,
                   'possible_bet':[1,3],
                   'initial_budget':100,
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
                 [params1,], #[params1, params2, params3, params4], 
                 num_simul=30, num_episodes=2000, eval_every=10000000000, 
                 alpha=0.0,
                 folder_path='./plot/tictactoe.png',
        )