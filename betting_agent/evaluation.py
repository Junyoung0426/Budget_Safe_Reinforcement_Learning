import numpy as np

def eval(env, Q_values, epi_to_eval):
    total_reward = 0.0
    for episode in range(epi_to_eval):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(Q_values[state]))
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
        total_reward += reward
    return total_reward / epi_to_eval