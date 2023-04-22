
import torch as t
import numpy as np
import torch.optim as optim
from torch.distributions import Beta

import sys
sys.path.append('..')

# Now you can import modules from the parent directory
from src.MLP import MLP

class PolicyGradientAgent:
    """
    """
    def __init__(self, env):

        # get lower and upper bounds from env, and store them for beta computation

        # instantiate the model as MLP
        self.state_shape = env.single_observation_space.shape[0]
        self.action_shape = env.single_action_space.shape[0]
        self.model = MLP([self.state_shape, 128, 128, 2*self.action_shape])

        # instantiates the normalizer module
        self.state_mean = t.zeros(self.state_shape)
        self.state_std = t.ones(self.state_shape)

        # instantiate the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def sample(self, obs):

        # convert obs to torch tensor
        obs_t = t.from_numpy(obs)
        if len(obs_t.size()) != 2:
            obs_t = obs_t.unsqueeze(0)

        # pass observation through normalizer
        obs_t = (obs_t-self.state_mean)/self.state_std

        # pass it through the network to get beta distribution parameters
        model_output = self.model(obs_t).detach()
        alphas = model_output[:, :self.action_shape]
        betas = model_output[:, -(self.action_shape+1):]

        # sample from beta distribution with those parameters
        dist = Beta(alphas, betas)

        return dist.sample()

    def gradient_step(self, trajs, only_return_grad=False):

        # assume trajs are already stored on gpu and in pytorch form

        # pass all observations through normalizer

        # pass trajectories through network, getting the beta parameters for all actions

        # compute log of beta distribution for all trajectories

        # compute importance sampling ratios and normalize them

        # subtract average return from returns in batch

        # define the loss as the sum over trajectories of the log probability scaled by the importance

        # differentiate that loss

        # take optim step

        # if only want to return grad, do that instead of taking an optim step

        raise NotImplemented

    def update_input_mean_std(self, mean, std):
        """
        update the mean and standard deviation of the distribution of inputs we expect
        """
        self.state_mean = mean
        self.state_std = std

    def update_average_return(self, average_return):
        """
        stores the average return of trajectories in buffer
        """
        self.average_return = average_return
