import numpy as np
#import torch
import pickle


dataset_path = f'fetch-expert-v2.pkl'

with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

# save all path information into separate lists

states, traj_lens, returns = [], [], []
actions = []
for path in trajectories:
    actions.append(np.min(path['actions']))
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum())
traj_lens, returns = np.array(traj_lens), np.array(returns)
print(len(trajectories))
print(np.mean(returns))
#print(np.max(returns))
#print(np.min(returns))
#print(np.std(returns))
print(np.mean(traj_lens))
print(states[1].shape)
print(np.max(actions))
print(np.min(actions))