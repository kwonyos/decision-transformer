import gym
import numpy as np
import torch
import pickle
from decision_transformer.models.decision_transformer import DecisionTransformer

def evaluate_episode_rtg_render(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):


    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # CHANGE
        state, reward, done, _ = env.step(action,1)
        env.render()

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

if __name__ == '__main__':
    mode = 'normal'
    num_eval_episodes = 100
    device = 'cuda'


    env = gym.make('gym_mujoco:ReacherTraj-v2')
    max_ep_len = 150
    env_targets = 76 # or 40
    scale = 10

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    dataset_path = f'data/trajreacher2d-expert-v2.pkl'

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6



     # CHANGE: load model, need to check
    model = torch.load('dt-agent.pkl')
    # model = model.model

    #evaluate_episode_rtg_render

    for _ in range(num_eval_episodes):
        with torch.no_grad():
            ret, length = evaluate_episode_rtg_render(
                env,
                state_dim,
                act_dim,
                model,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=env_targets/scale,
                mode=mode,
                state_mean=state_mean,
                state_std=state_std,
                device=device,
            )