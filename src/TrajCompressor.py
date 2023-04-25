
import pickle
import gymnasium as gym
from utils import *
import numpy as np
import torch as t
from agents.PolicyGradientAgent import *
from torch.autograd import Variable
from ReplayBuffer import *

class TrajCompressor:
    """
    """
    def __init__(self, history_dict, env_name, num_envs=8):

        self.env_name = env_name
        self.env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=True)
        self.single_env = gym.make(env_name, render_mode='rgb_array')
        self.history_dict = history_dict

    def find_cosine_distance_of_sequence(self, agent, target_w, state_seq, action_seq, init_alpha=None):
        """
        given an agent, a target direction in its weight space, a given sequence of states and sequence of actions,
        find the cosine similarity of the gradients engendered by the state-action sequence and the target weight direction.

        find_closest_ray_in_vector_span actually find the best vector in the whole span of the gradients for each action
        in the sequence.

        We return the cosine similarity and the coefficients by which we need to multiply the action sequence gradients
        in order to maximize this cosine similarity
        """

        model_gradients = agent.get_model_gradient_for_each_action(state_seq, action_seq)
        cosine_similarity, alphas, closest_grad = find_closest_ray_in_vector_span(target_w, model_gradients, init_alpha=init_alpha)

        return cosine_similarity, alphas, closest_grad

    def find_compressed_grad_trajectory(self, agent, target_w, traj_length, verbose=False, init_action_seq=None, init_alpha=None):
        """
        should find a trajectory of length l which make it so that they produce
        gradients very close to the true policy gradient, or any other desired weight vector.

        If we have a target vector w_target, and a sequence of gradients w_i, we want the coefficients a_i such that

        w_target dot (sum_i a_i * w_i)/ |sum_i a_i * w_i| is maximised, we need gradient descent to do this.

        """
        prev_action_seq = [] if init_action_seq is None else init_action_seq
        length_to_optim = traj_length - len(init_action_seq) if init_action_seq is not None else traj_length

        decrease_std_after_n_not_improved = 5
        not_improved_counter = 0

        # sample initial sequence:
        action_explore_std = 0.01
        print("randomizing actions")
        # this sequence is what we actually optimise
        action_seq = [action_explore_std * self.single_env.action_space.sample() for i in range(length_to_optim)]

        if init_alpha is not None:
            init_alpha = t.cat([init_alpha, t.zeros(traj_length - len(init_alpha))])

        state_seq = get_state_trajectory_from_action_seq(prev_action_seq + action_seq, self.single_env, seed=42)
        cosine_similarity, alphas, closest_grad = \
            self.find_cosine_distance_of_sequence(agent, target_w, state_seq, prev_action_seq + action_seq, init_alpha=init_alpha)

        print(f"cos similarity before optimizing: {cosine_similarity:.7f}")

        for i in range(100):

            new_action_seq = [x + action_explore_std * np.random.randn(*x.shape) for x in action_seq]

            new_concated_seq = prev_action_seq + new_action_seq

            new_state_seq = get_state_trajectory_from_action_seq(new_concated_seq, self.single_env, seed=42)
            new_cosine_similarity, new_alphas, new_closest_grad = \
                self.find_cosine_distance_of_sequence(agent, target_w, new_state_seq, new_concated_seq, init_alpha=alphas)

            if verbose:
                print(f"{i}:new cosine sim: {new_cosine_similarity:.3f} best cosine sim:{cosine_similarity:.7f}"
                      f", std:{action_explore_std:.3e}, traj_len:{len(new_concated_seq)}")

            if new_cosine_similarity > cosine_similarity:
                action_seq = new_action_seq
                state_seq = new_state_seq
                cosine_similarity = new_cosine_similarity
                alphas = new_alphas
                closest_grad = new_closest_grad

                not_improved_counter = 0
            else:
                not_improved_counter += 1

            if not_improved_counter > decrease_std_after_n_not_improved:
                action_explore_std *= 0.5
                not_improved_counter = 0

        return (prev_action_seq + action_seq), state_seq, closest_grad, cosine_similarity, alphas


    # NOT USED RIGHT NOW
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

if __name__ == "__main__":

    exploration_std = 0.1
    device = t.device('cpu')

    history_dict = pickle.load(open('../data/quad_Ant-v4_3/run_dict', 'rb'))

    compressor = TrajCompressor(history_dict, history_dict['env_name'])

    agent = PolicyGradientAgent(env=compressor.env, exploration_std=exploration_std, device=device , type_model=history_dict['type_model'])
    agent.model.load_state_dict(history_dict['model_dict'][-1])
    agent.optimizer = optim.SGD(agent.model.parameters(), lr=0.001)

    # =======================================================
    # need to recompute state means and variances because I didn't save them in the dict...
    # =======================================================
    print(f"computing initial state means and variances")
    trajs = sample_trajectories(agent, compressor.env, 100, 100, random_sampling=True)
    buffer = ReplayBuffer()
    buffer.add_trajectories(trajs)
    data_mean, data_std = buffer.get_data_mean_std()
    agent.update_input_mean_std(data_mean, data_std)
    buffer.buffer = []

    # =======================================================
    trajs = sample_trajectories(agent, compressor.env, 1, max_steps=1000, seed=42)
    traj = trajs[0]
    agent_actions = list(trajs[0]['a'])
    # =======================================================


    # the difference in weights from the initial values to the last weight values
    target_w = flatten_params(list(history_dict['model_dict'][-1].values())) -\
               flatten_params(list(history_dict['model_dict'][0].values()))

    print(f"size of target parameters is {target_w.size()}")

    # finding the best action sequence which matches the difference in weights

    # action_seq = agent_actions[:50]
    action_seq = None
    alphas = None

    for traj_length in range(50, 500, 10):

        action_seq, state_seq, closest_grad, cosine_similarity, alphas =\
            compressor.find_compressed_grad_trajectory(agent, target_w=target_w,
                                                       traj_length=traj_length,
                                                       init_action_seq=action_seq,
                                                       verbose=True,
                                                       init_alpha=alphas)

        alphas = alphas.detach()


        get_video_from_env(compressor.single_env, action_seq,
                           save_path='../data/compressor_videos/' + "length_" +
                                     str(traj_length) + f"_cosSim_{cosine_similarity:.3f}" + ".mp4",
                           seed=42, alphas=alphas.numpy())
