

import numpy as np
import torch as t
from torch.autograd import Variable

class TrajCompressor:
    """
    """
    def __init__(self):
        pass

    def find_compressed_grad_trajectory(self):
        """
        should find the set of k trajectories of length l which make it so that they produce
        gradients very close to the true policy gradient, or any other desired weight vector
        """
        pass

    def find_model_policy_gradient(self):
        """
        samples from the model with a very large batch size and finds the full policy gradient
        """
        pass

    def sample_actions_gaussian(self, env, n_samples, mean, std_dev, n_traj_length):
        """
        randomly samples actions sequences of given length within the action space of env
        """
        pass

    def compute_traj_grad(self, model, trajs):
        """
        computes the gradient with respect to the model for all trajectories
        """
        pass

    def evaluate_action_sequences_on_env(self,):
        """
        evaluates a bunch of different action sequences of the same length on an environment wrapped in
        Async Vector Env to get sequence of states
        """
        pass

    def CEM(self, solution_sampler, init_std, eval_fn, max_iter, n_sample_per_iter, elite_frac=0.3):
        """
        implements the Cross-Entropy-Method on an unknown function f(x), using exclusively pytorch data types and
        methods.
        Arguments:
            solution_sampler: function which samples new solutions with given mena and variance
            init_std: the initial standard deviation of our solutions
            eval_fn: the function we want to optimize, used to evaluate solutions
            max_iter: the maximum number of iterations CEM should run for
            n_sample_per_iter: the number of samples per iteration
            elite_frac: the number of elites to keep at each iteration
        """
        # Determine the number of elite samples
        n_elite = int(n_sample_per_iter * elite_frac)

        # Initialize mean and standard deviation
        mean = t.zeros_like(init_std)
        std = init_std.clone()

        for _ in range(max_iter):
            # Sample solutions
            solutions = solution_sampler(mean, std, n_sample_per_iter)

            # Evaluate solutions
            scores = eval_fn(solutions)

            # Sort solutions by scores in descending order
            _, top_indices = t.sort(scores, descending=True)

            # Select the elite samples
            elite_solutions = solutions[top_indices[:n_elite]]

            # Update the mean and standard deviation
            mean = t.mean(elite_solutions, dim=0)
            std = t.std(elite_solutions, dim=0, unbiased=False)

        return mean


