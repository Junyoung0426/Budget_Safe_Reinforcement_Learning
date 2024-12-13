
from agent import *
from simulate import *
from environment import *
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=4)
args = parser.parse_args()

params1 = {'algorithm':'Aggressive',
        'eps':0.1, 'eps_decay':(100, 0.9), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
        'beta':(0.0, 0.0), # ucb constant
        'learning_rate':lambda epi:(1.0)/max(1,epi), # decay learning rate
        'betting_strategy':'max', 'threshold':0,
        'possible_bet':[1,3],
        'initial_budget':300,
        'dynamic_betting':False,
        'dynamic_possible_bet':None,
}

params2 = deepcopy(params1)
params2['algorithm']='Conservative'
params2['betting_strategy'] = 'min'

simulate('tictactoe',
        [params1, params2],
        num_simul=30, num_episodes=3000, eval_every=10000000000,
        ci_alpha=0.0,
        line_alpha = [1.0, 1.0],
        folder_path='./additional_figures/eval_motivation_fig1/',
        only_plot=True
)



params1 = {'algorithm':'Aggressive',
        'eps':0.1, 'eps_decay':(100, 0.9), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
        'beta':(0.0, 0.0), # ucb constant
        'learning_rate':lambda epi:(1.0)/max(1,epi), # decay learning rate
        'betting_strategy':'max', 'threshold':0,
        'possible_bet':[1,3],
        'initial_budget':300,
        'dynamic_betting':False,
        'dynamic_possible_bet':None,
}

params2 = deepcopy(params1)
params2['algorithm']='Conservative'
params2['betting_strategy'] = 'min'

simulate('tictactoe',
        [params1, params2],
        num_simul=30, num_episodes=3000, eval_every=10000000000,
        ci_alpha=0.0,
        line_alpha = [1.0, 0.2],
        folder_path='./additional_figures/eval_motivation_fig2/',
        only_plot=True
)


params1 = {'algorithm':'Aggressive',
        'eps':0.1, 'eps_decay':(100, 0.9), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
        'beta':(0.0, 0.0), # ucb constant
        'learning_rate':lambda epi:(1.0)/max(1,epi), # decay learning rate
        'betting_strategy':'max', 'threshold':0,
        'possible_bet':[1,3],
        'initial_budget':300,
        'dynamic_betting':False,
        'dynamic_possible_bet':None,
}

params2 = deepcopy(params1)
params2['algorithm']='Conservative'
params2['betting_strategy'] = 'min'

simulate('tictactoe',
        [params1, params2],
        num_simul=30, num_episodes=3000, eval_every=10000000000,
        ci_alpha=0.0,
        line_alpha = [0.2, 1.0],
        folder_path='./additional_figures/eval_motivation_fig3/',
        only_plot=True
)


params1 = {'algorithm':'Aggressive',
        'eps':0.1, 'eps_decay':(100, 0.9), # decay epsilon every eps_decay[0] by eps = eps * eps_decay[1]
        'beta':(0.0, 0.0), # ucb constant
        'learning_rate':lambda epi:(1.0)/max(1,epi), # decay learning rate
        'betting_strategy':'max', 'threshold':0,
        'possible_bet':[1,3],
        'initial_budget':300,
        'dynamic_betting':False,
        'dynamic_possible_bet':None,
}

params2 = deepcopy(params1)
params2['algorithm']='Conservative'
params2['betting_strategy'] = 'min'

params3 = deepcopy(params1)
params3['algorithm']='Our goal'
params3['betting_strategy'] = 'lcb'

simulate('tictactoe',
        [params1, params2, params3],
        num_simul=30, num_episodes=3000, eval_every=10000000000,
        ci_alpha=0.0,
        line_alpha = [0.2, 0.2, 1.0],
        folder_path='./additional_figures/eval_motivation_fig4/',
        only_plot=True
)