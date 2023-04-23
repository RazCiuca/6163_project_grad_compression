
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
    def __init__(self, env_name, num_envs=64):

        #  environment name
        self.env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=True)
        #  replay buffer which stores the trajectories
        self.buffer = ReplayBuffer()
        # instantiate the agent, the optimizer is stored in the agent
        self.agent = PolicyGradientAgent(self.env)

    def train(self, n_samples, n_iterations, traj_batch_size, max_steps=1000, verbose=True,
              sample_randomly_instead_of_agent=False):

        # sample new trajectories from the current agent and add to buffer
        trajs = sample_trajectories(self.agent, self.env, n_samples, max_steps,
                                    random_sampling=sample_randomly_instead_of_agent)
        # add to buffer
        self.buffer.add_trajectories(trajs)

        # compute input means and variance of current states
        data_mean, data_std = self.buffer.get_data_mean_std()
        average_return = self.buffer.get_average_return()
        self.agent.update_input_mean_std(data_mean, data_std)
        self.agent.update_average_return(average_return)


        # todo: call ReplayBuffer to send everything to gpu and have it stored in pytorch

        # for loop which samples from the replay buffer and does SGD
        for i in range(n_iterations):
            if i%100 == 0:
                print(f"training iteration: {i}")

            # sample trajectories from buffer
            # obs, actions, mean_rewards, log_probs, sample_indices
            obs, actions, mean_rewards, log_probs, sample_indices = self.buffer.sample(traj_batch_size)
            trajs_batch = (obs, actions, mean_rewards, log_probs, sample_indices)
            IS_ratios = self.agent.gradient_step(trajs_batch)

            self.buffer.update_IS_ratios(IS_ratios, sample_indices)

        # remove trajectories which have low importance sampling ratios, which are now useless for our trainer
        self.buffer.purge_low_IS_ratios()

        return average_return
