

import pickle
import gymnasium as gym
from utils import *
import numpy as np
import torch as t
from agents.PolicyGradientAgent import *
from torch.autograd import Variable
from ReplayBuffer import *
from TrajCompressor import *

"""
given a compressed trajectory, move in that direction
"""
if __name__ == "__main__":

    # load the base model, the trajectory and the gradient direction
    exploration_std = 0.1
    device = t.device('cpu')
    history_dict = pickle.load(open('../data/net_Ant-v4_0/run_dict', 'rb'))

    # loading the agent with the initial checkpoint weights
    compressor = TrajCompressor(history_dict, history_dict['env_name'])
    agent = PolicyGradientAgent(env=compressor.env, exploration_std=exploration_std, device=device,
                                type_model=history_dict['type_model'])
    agent.model.load_state_dict(history_dict['model_dict'][0])
    agent.optimizer = optim.SGD(agent.model.parameters(), lr=0.001)

    # loading the optimally compressed action sequence
    action_seq = pickle.load(open('../data/compressor_video_ant_net/action_seqs/dict_490', 'rb'))
    search_dir = action_seq['closest_grad']
    search_dir /= t.norm(search_dir)

    # this is the difference from the final weights to the original values
    init_w = flatten_params(list(history_dict['model_dict'][0].values()))
    target_w = flatten_params(list(history_dict['model_dict'][-1].values())) - \
               init_w

    weight_norm = t.norm(target_w)

    # define the magnitude of weight change by looking at the magnitude of weight change with the final trained network

    # sampled loads of trajectories with a small explore_std to find their rewards

    x_data = np.arange(0.2, 0.6, 0.05)
    y_data = []

    for beta in x_data:

        new_weights = init_w + beta * weight_norm * search_dir
        new_weights = unflatten_params(new_weights, list(agent.model.parameters()))

        # load weights into model
        for w, new_w in zip(agent.model.parameters(), new_weights):
            w.data = new_w

        buffer = ReplayBuffer()
        trajs = sample_trajectories(agent, compressor.env, n_traj=300, max_steps=200, random_sampling=False, explore_std=0.05)
        buffer.add_trajectories(trajs)
        average_return = buffer.get_average_return()
        print(f"beta:{beta:.4f}, average_return:{average_return:.3f}")
        y_data.append(average_return)

        video_filename = "../data/compressor_line_search_ant_net_videos/"
        video_traj = sample_trajectories(agent, compressor.env, n_traj=1, max_steps=1000, random_sampling=False, explore_std=0.05)[0]
        actions = video_traj['a']
        get_video_from_env(compressor.single_env, actions, video_filename + f"beta_{beta:.2f}_rew_{average_return:.4f}" + ".mp4")

    pickle.dump([x_data, y_data], open('../data/ant_quad_return_line_search_data', 'wb'))