
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MLP import *
import gymnasium as gym
import imageio

def get_video_from_env(env, actions, save_path=None):
    """
    Get a video from the environment using the given actions.

    :param env: The environment to get the video from.
    :param actions: The actions to use to get the video.
    :return: The video as a list of images.
    """
    video = []
    observation, info = env.reset()
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        video.append(env.render(mode='rgb_array'))
        if terminated or truncated:
            break

    if save_path is not None:
        imageio.mimsave(save_path, video)

    return video

def sample_trajectories(agent, env, n_traj=1000, max_steps=1000):
    """
    Sample trajectories from the environment using the model.

    :param model: The model to use to sample the trajectories.
    :param env: The environment to sample the trajectories from, assuming vectorized.
    :return: A list of trajectories. Each trajectory is a list of dictionaries with keys 'obs', 'a', 'r'.
    """
    trajectories = []
    obs_traj = []
    a_traj = []
    r_traj = []

    for traj in range(n_traj):
        obs, info = env.reset()

        traj_log_prob = 0

        for step in range(max_steps):
            action, action_log_prob = agent.sample(obs)  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            obs_traj.append(observation)
            a_traj.append(action)
            r_traj.append(reward)
            traj_log_prob += action_log_prob

            if terminated or truncated or step == max_steps-1:
                trajectories.append({'obs': np.stack(obs_traj, axis=0),
                                     'a': np.stack(a_traj, axis=0),
                                     'r': np.stack(r_traj, axis=0),
                                     'sampling_log_prob': traj_log_prob})
                obs_traj = []
                a_traj = []
                r_traj = []
                break

    return trajectories

