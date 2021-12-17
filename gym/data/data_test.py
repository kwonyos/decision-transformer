import numpy as np
#import torch
import pickle


dataset_path = f'fetch-expert-v2.pkl'

with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

# save all path information into separate lists

states, traj_lens, returns = [], [], []
for path in trajectories:
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum())
traj_lens, returns = np.array(traj_lens), np.array(returns)
print(np.mean(returns))
#print(np.max(returns))
#print(np.min(returns))
#print(np.std(returns))
print(np.min(traj_lens))
print(states[1].shape)