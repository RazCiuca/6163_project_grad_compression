
import torch as t
from torch.distributions import beta
import torch.nn as nn
from utils import *
from replay_buffer import *
from MLP import *

class RLagent:
    """
    this contains the MLP weights, the optimizer, and sampling routines with beta distributions
    """
    def __init__(self, obs_shape, action_shape):
        self.action_shape = action_shape
        self.mlp = MLP([obs_shape, 128, 256, action_shape*2])
        self.optimizer = optim.SGD(self.mlp.parameters(), lr=0.01, momentum=0.9)
        self.importance_ratio_threshold = 0.005

    def train(self, replay_buffer, n_iter, batch_size):

        for iter in range(n_iter):
            # sample from replay buffer
            obs, actions, mean_rewards, log_probs, sample_indices = replay_buffer.sample(batch_size)
            indices_to_remove = []

            average_mean_reward = t.mean(t.tensor(mean_rewards))

            # send model and data to gpu

            # compute beta log likelihood weighted by mean_rewards and importance ratio

            loss = 0

            for observation, action, mean_reward, log_prob, index in \
                    zip(obs, actions, mean_rewards, log_probs, sample_indices):
                # this is the agent output for every observation in the trajectory

                # computing the beta distribution that the agent currently assigns to every observation in the traj.
                agent_output = self.mlp.forward(observation)
                agent_output = t.exp(agent_output.view(-1, self.action_shape, 2))
                m = beta.Beta(agent_output[:, :, 0], agent_output[:, :, 1])

                # this is the log probability that the agent assigns to current action sequence under consideration
                agent_log_prob = m.log_prob((action+1.0)/2.0)

                # the importance sampling ratio for the trajectory, we need to detach to avoid backproping through this
                importance_ratio = np.exp((agent_log_prob.detach() - log_prob).sum())

                if importance_ratio < self.importance_ratio_threshold:
                    indices_to_remove.append(index)

                # when this gets differentiated, it will compute the negative policy gradient with a baseline
                loss -= (mean_reward - average_mean_reward) * importance_ratio * agent_log_prob

            # remove from buffer data with importance ratios less than some number
            replay_buffer.remove_indices_from_buffer(indices_to_remove)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def sample(self, observation):
        """
        here observation is a numpy array
        returns a sample from the beta distribution defined by the model, as well as the log likelihood of the sample
        """
        with t.no_grad():
            x = self.mlp.forward(t.from_numpy(observation).float()[None])
            x = t.exp(x.view(-1, self.action_shape, 2))

            m = beta.Beta(x[:, :, 0], x[:, :, 1])
            actions = m.sample()
            log_probs = m.log_prob(actions)

            return actions * 2 - 1, log_probs

