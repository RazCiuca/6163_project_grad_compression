
import torch as t
import numpy as np
import torch.nn as nn


class ReplayBuffer:
    """
    class implementing a replay buffer, keeping track of importance sampling ratios

    todo: need to gracefully treat trajectories with different lengths

    """
    def __init__(self):
        self.buffer = []

    def add_trajectories(self, trajs):
        """
        trajs is a list of dict with keys "obs", "rewards", "actions", "sampling_log_prob", "importance_ratio"
        """
        self.buffer += trajs

    def sample(self, n_traj):
        """
        returns a list of trajectories, mean rewards and sampling_log_probs to be passed to the training loop
        """

        sample_indices = np.random.choice(len(self.buffer), n_traj) \
            if n_traj < len(self.buffer) else np.arange(len(self.buffer))

        obs = [t.tensor(self.buffer[index]['obs'], dtype=t.float32) for index in sample_indices]
        actions = [t.tensor(self.buffer[index]['a'], dtype=t.float32)for index in sample_indices]
        mean_rewards = [t.tensor(self.buffer[index]['r'], dtype=t.float32).mean() for index in sample_indices]
        log_probs = [self.buffer[index]['sampling_log_prob'] for index in sample_indices]

        return obs, actions, mean_rewards, log_probs, sample_indices

    def remove_indices_from_buffer(self, indices):
        """
        removes the given indices from the buffer
        """
        for x in sorted(indices)[::-1]:
            self.buffer.pop(x)

    def update_IS_ratios(self, IS_ratios, sample_indices):
        """
        updates the ['importance_ratio'] field of trajectories at given sample_indices
        """
        for index, ratio in zip(sample_indices, IS_ratios):
            self.buffer[index]['importance_ratio'] = ratio

    def purge_low_IS_ratios(self, threshold=1e-7):
        """
        removes from list all indices with IS_ratios below the threshold
        """
        indices_to_remove = []
        for i in range(len(self.buffer)):
            if 'importance_ratio' in self.buffer[i]:
                if self.buffer[i]['importance_ratio'] < threshold:
                    indices_to_remove.append(i)

        self.remove_indices_from_buffer(indices_to_remove)

    def get_data_mean_std(self):
        """
        return the mean and variance over the states currently in the buffer
        """
        state_mean = 0
        for traj in self.buffer:
            state_mean += traj['obs'].mean(axis=0)

        state_mean /= len(self.buffer)

        state_var = 0
        for traj in self.buffer:
            state_var += ((traj['obs']-state_mean)**2).mean(axis=0)

        state_std = (state_var/len(self.buffer))**0.5

        return state_mean, state_std

    def get_average_return(self):
        """
        computes the average mean_rewards of trajectories in the dataset
        """
        total_return = 0
        for traj in self.buffer:
            total_return += traj['r'].sum()

        total_return /= len(self.buffer)

        return total_return

