
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
        self.action_high_bound = t.tensor(env.single_action_space.high)
        self.action_low_bound = t.tensor(env.single_action_space.low)

        # instantiate the model as MLP
        self.state_shape = env.single_observation_space.shape[0]
        self.action_shape = env.single_action_space.shape[0]
        self.model = MLP([self.state_shape, 128, 128, 2*self.action_shape])

        # instantiates the normalizer module
        self.state_mean = t.zeros(self.state_shape)
        self.state_std = t.ones(self.state_shape)
        self.average_return = 0
        # instantiate the optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.9)

    def sample(self, obs):
        """
        here obs is a numpy array
        """

        # convert obs to torch tensor
        obs_t = t.from_numpy(obs).float()
        if len(obs_t.size()) != 2:
            obs_t = obs_t.unsqueeze(0)

        # pass observation through normalizer
        obs_t = (obs_t-self.state_mean)/self.state_std

        # pass it through the network to get beta distribution parameters
        model_output = self.model(obs_t).detach()
        alphas = model_output[:, :self.action_shape]
        betas = model_output[:, -self.action_shape:]

        # print(model_output.mean(0))

        # sample from beta distribution with those parameters
        dist = Beta(alphas, betas)

        new_sample = dist.sample()
        # print(new_sample.size())

        log_prob = dist.log_prob(new_sample).sum(dim=1)

        return self.action_low_bound + (self.action_high_bound-self.action_low_bound) * new_sample.numpy(), log_prob.numpy()

    def gradient_step(self, trajs, only_return_grad=False):
        """
        assume trajs are already stored on gpu and in pytorch form
        """

        loss = 0

        # the log of
        log_IS_ratios = []
        obs_, actions_, mean_rewards_, sampling_log_probs_, sample_indices_ = trajs

        for obs, actions, mean_rewards, sampling_log_probs, sample_indices in \
                zip(obs_, actions_, mean_rewards_, sampling_log_probs_, sample_indices_):

            # pass all observations through normalizer obs has shape [max_traj_length, obs_space]
            obs = (obs-self.state_mean)/self.state_std

            # pass trajectories through network, getting the beta parameters for all actions
            model_output = self.model(obs)
            alphas = model_output[:, :self.action_shape]
            betas = model_output[:, -(self.action_shape):]
            dist = Beta(alphas, betas)
            # compute log of beta distribution for all trajectories
            # remembering that actions are stored between self.action_low_bound and self.action_high_bound
            log_probs = dist.log_prob((actions-self.action_low_bound)/(self.action_high_bound-self.action_low_bound))

            IS_ratio = t.exp(log_probs.detach().sum() - sampling_log_probs)
            log_IS_ratios.append(IS_ratio)

            loss += - (mean_rewards - self.average_return) * IS_ratio * log_probs.mean()

        loss /= sum(log_IS_ratios)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # todo: if only want to return grad, do that instead of taking an optim step

        return log_IS_ratios


    def update_input_mean_std(self, mean, std):
        """
        update the mean and standard deviation of the distribution of inputs we expect
        """
        self.state_mean = t.tensor(mean).float()
        self.state_std = t.tensor(std).float()

    def update_average_return(self, average_return):
        """
        stores the average return of trajectories in buffer
        """
        self.average_return = average_return
