
import torch as t
import numpy as np
import torch.optim as optim
from torch.distributions import Beta, Normal

import sys
sys.path.append('..')

# Now you can import modules from the parent directory
from src.MLP import *

class PolicyGradientAgent:
    """
    """
    def __init__(self, env, exploration_std, device, lr=None, type_model='quad'):

        self.device = device

        # get lower and upper bounds from env, and store them for beta computation
        self.action_high_bound = t.tensor(env.single_action_space.high, device=device)
        self.action_low_bound = t.tensor(env.single_action_space.low, device=device)

        # instantiate the model as MLP
        self.state_shape = env.single_observation_space.shape[0]
        self.action_shape = env.single_action_space.shape[0]

        # NOT USED: this will make it so the initial standard deviation of the policy will match init_exploration_std
        # model_beta = np.log(2)/((init_exploration_std**(-2) - 4.0) / 8.0)

        # self.model = MLP([self.state_shape, 128, self.action_shape])
        # self.model.set_output_std(t.randn(1000, self.state_shape), out_std=0.1)
        if type_model == 'quad':
            self.model = MultidimensionalQuadraticRegression(self.state_shape, self.action_shape, order=2)
        elif type_model == 'cube':
            self.model = MultidimensionalQuadraticRegression(self.state_shape, self.action_shape, order=3)
        elif type_model == 'net':
            self.model = MLP([self.state_shape, 128, self.action_shape])
            self.model.set_output_std(t.randn(1000, self.state_shape), out_std=0.1)

        self.model = self.model.to(device)
        self.explore_std = exploration_std

        # instantiates the normalizer module
        self.is_mean_initialized = False
        self.state_mean = t.zeros(self.state_shape, device=self.device)
        self.state_std = t.ones(self.state_shape, device=self.device)
        self.average_return = 0
        # instantiate the optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=(lr if lr is not None else 1e-4), momentum=0.9)


    def sample(self, obs, veto_explore_std=None):
        """
        here obs is a numpy array
        """

        # convert obs to torch tensor
        obs_t = t.from_numpy(obs).float().to(self.device)
        if len(obs_t.size()) != 2:
            obs_t = obs_t.unsqueeze(0)

        # pass observation through normalizer
        obs_t = (obs_t-self.state_mean)/self.state_std

        # pass it through the network to get beta distribution parameters
        action_means = self.model(obs_t).detach()
        dist = Normal(action_means, (self.explore_std if veto_explore_std is None else veto_explore_std))
        new_sample = dist.sample()
        # print(new_sample.size())
        log_prob = dist.log_prob(new_sample).sum(dim=1)

        return new_sample.cpu().numpy(), log_prob.cpu().numpy()

    def gradient_step(self, trajs, only_return_grad=False, print_grad_norm=True):
        """
        assume trajs are already stored on gpu and in pytorch form
        """

        loss = 0

        self.optimizer.zero_grad()

        # importance sampling ratios
        IS_ratios = []
        obs_, actions_, sum_rewards_, sampling_log_probs_, sample_indices_ = trajs

        # ===================== ALL FORWARD PASSES =====================

        actions_means = [self.model((obs-self.state_mean)/self.state_std) for obs in obs_]

        for action_mean, actions, sum_rewards, sampling_log_probs in \
                zip(actions_means, actions_, sum_rewards_, sampling_log_probs_):

            # print(obs.mean(dim=0))
            dist = Normal(action_mean, self.explore_std)
            # print(action_means.mean(), action_means.std())
            # compute log of prob for all trajectories
            log_probs = dist.log_prob(actions)

            IS_ratio = t.exp(log_probs.detach().sum() - sampling_log_probs)

            IS_ratios.append(IS_ratio)

            loss += - (sum_rewards - self.average_return) * IS_ratio * log_probs.mean()
            # loss += - (mean_rewards - self.average_return) * log_probs.mean()/len(obs_)

        loss /= sum(IS_ratios)
        loss.backward()
        total_norm_before_clip = t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        IS_ratios = [x.cpu().item() for x in IS_ratios]

        if print_grad_norm:
            print(f"gradient norm = {total_norm_before_clip.cpu().item()}, "
                  f"min IS: {np.min(IS_ratios)}, "
                  f"max IS: {np.max(IS_ratios)}, "
                  f"ave IS:{np.mean(IS_ratios)}")

        # todo: if only want to return grad, do that instead of taking an optim step

        return IS_ratios


    def update_input_mean_std(self, mean, std):
        """
        update the mean and standard deviation of the distribution of inputs we expect
        """
        gamma = 0.9
        if not self.is_mean_initialized:
            self.state_mean = t.tensor(mean, device=self.device).float()
            self.state_std = t.tensor(std, device=self.device).float()
            self.is_mean_initialized = True
        else:
            self.state_mean = gamma * self.state_mean + (1-gamma) * t.tensor(mean, device=self.device).float()
            self.state_std = gamma * self.state_std + (1-gamma) * t.tensor(std, device=self.device).float()

    def update_average_return(self, average_return):
        """
        stores the average return of trajectories in buffer
        """
        self.average_return = average_return
