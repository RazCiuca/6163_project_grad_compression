
import pickle
import copy
import torch as t
import torch.optim as optim
from utils import *
import gymnasium as gym
from ReplayBuffer import ReplayBuffer
from agents.PolicyGradientAgent import PolicyGradientAgent

class Trainer:
    """
    - stores the environment, given the env name
    - stores the ReplayBuffer
    - stores the Agent instance
    - does the env_interact -> SGD_loop_through_data -> env_interact macro loop
    """
    def __init__(self, exp_dict, env_name, exp_name, type_model, num_envs=64, init_exploration_std=0.1, device=t.device('cuda'), lr=None):

        self.run_dict = { 'exp_dict': exp_dict,
                          'env_name': env_name,
                          'exp_name': exp_name,
                          'type_model': type_model,
                          'model_dict': [],
                          'mean_reward': []}

        self.exp_name = exp_name

        self.device = device
        #  environment name
        self.env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=True)
        self.single_env = gym.make(env_name, render_mode='rgb_array')
        #  replay buffer which stores the trajectories
        self.buffer = ReplayBuffer()
        # instantiate the agent, the optimizer is stored in the agent
        self.agent = PolicyGradientAgent(self.env, init_exploration_std, device=device, lr=lr, type_model=type_model)

    def save_state(self):
        # save the dict with pickle
        pickle.dump(self.run_dict, open('../data/' + self.run_dict['exp_name'] + '/run_dict', 'wb'))

    def get_initial_state_mean_variance(self):
        print(f"computing initial state means and variances")
        trajs = sample_trajectories(self.agent, self.env, 100, 100, random_sampling=True)
        self.buffer.add_trajectories(trajs)
        data_mean, data_std = self.buffer.get_data_mean_std()
        self.agent.update_input_mean_std(data_mean, data_std)
        self.buffer.buffer = []

    def train(self, n_samples, n_iterations, max_IS, traj_batch_size, max_steps=1000, verbose=True,
              sample_randomly_instead_of_agent=False, save_video_name=None):

        # sample new trajectories from the current agent and add to buffer
        trajs = sample_trajectories(self.agent, self.env, n_samples, max_steps,
                                    random_sampling=sample_randomly_instead_of_agent)

        self.buffer.buffer = []
        # add to buffer
        self.buffer.add_trajectories(trajs)

        average_return = self.buffer.get_average_return()
        self.agent.update_average_return(average_return)

        # data_mean, data_std = self.buffer.get_data_mean_std()
        # self.agent.update_input_mean_std(data_mean, data_std)

        self.run_dict['mean_reward'].append(average_return)

        if save_video_name is not None:
            visual_traj = sample_trajectories(self.agent, self.env, 1, 1000, random_sampling=False, explore_std=0.05)
            traj = visual_traj[0]
            actions = traj['a']
            get_video_from_env(self.single_env, actions, save_video_name + f"_rew_{average_return:.4f}" + ".mp4")

        # for loop which samples from the replay buffer and does SGD
        for i in range(n_iterations):
            if i%10 == 0:
                print(f"training iteration: {i}")

            # sample trajectories from buffer
            # obs, actions, mean_rewards, log_probs, sample_indices

            # ====================================== SAMPLING ======================================
            # start = t.cuda.Event(enable_timing=True)
            # end = t.cuda.Event(enable_timing=True)
            # start.record()
            obs, actions, sum_rewards, log_probs, sample_indices = self.buffer.sample(traj_batch_size, device=self.device)
            # end.record()
            # t.cuda.synchronize()
            #
            # if i%100==0: print(f"sampling time= {start.elapsed_time(end)}")

            trajs_batch = (obs, actions, sum_rewards, log_probs, sample_indices)

            # ====================================== TRAINING ======================================
            # start = t.cuda.Event(enable_timing=True)
            # end = t.cuda.Event(enable_timing=True)
            # start.record()
            IS_ratios = self.agent.gradient_step(trajs_batch, print_grad_norm=(i%10 == 0))
            # end.record()
            # t.cuda.synchronize()
            # if i % 100 == 0: print(f"training time= {start.elapsed_time(end)}")

            self.buffer.update_IS_ratios(IS_ratios, sample_indices)
            # remove trajectories which have low importance sampling ratios, which are now useless for our trainer
            buffer_size, current_max_IS = self.buffer.purge_low_IS_ratios()

            # update average return weighted by the importance ratio
            average_return = self.buffer.get_average_return()
            self.agent.update_average_return(average_return)
            print(f"average return: {average_return:.4f}")

            # if we have less trajectories
            if current_max_IS > max_IS or len(self.buffer.buffer) <= traj_batch_size/2:
                print('hit max allowed IS ratio')
                break

        # compute input means and variance of current states
        # NOT IF WE DO POLYNOMIAL REGRESSION
        # data_mean, data_std = self.buffer.get_data_mean_std()
        # self.agent.update_input_mean_std(data_mean, data_std)

        print(f"buffer_size={len(self.buffer.buffer)}")
        self.run_dict['model_dict'].append(copy.deepcopy(self.agent.model.state_dict()))

        return average_return

