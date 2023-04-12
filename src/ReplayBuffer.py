
import torch as t
import numpy as np
import torch.nn as nn


class ReplayBuffer:
    """
    class implementing a replay buffer, keeping track of importance sampling ratios
    """
    def __init__(self):
        self.buffer = []

    def add_trajectories(self, trajs):
        """
        trajs is a list of dict with keys "obs", "rewards", "actions", "sampling_log_prob", "importance_ratio"
        """
        self.buffer += trajs

    def sample(self, n_traj):
        # returns a list of trajectories, mean rewards and sampling_log_probs to be passed to the training loop

        sample_indices = np.random.choice(len(self.buffer), n_traj) \
            if n_traj < len(self.buffer) else np.arange(len(self.buffer))

        obs = [t.tensor(self.buffer[index]['obs']).float() for index in sample_indices]
        actions = [t.tensor(self.buffer[index]['a']).float() for index in sample_indices]
        mean_rewards = [t.tensor(self.buffer[index]['rewards']).float().mean() for index in sample_indices]
        log_probs = [self.buffer[index]['sampling_log_prob'] for index in sample_indices]

        return obs, actions, mean_rewards, log_probs, sample_indices

    def remove_indices_from_buffer(self, indices):

        for x in indices.sort()[::-1]:
            self.buffer.pop(x)

