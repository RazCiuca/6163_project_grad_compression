import numpy as np

from MLP import *
import imageio

def unflatten_params(flat_tensor_param, like_params):
    """
    takes a flat_tensor_param and produces
    """
    sizes = [np.prod(list(x.size())) for x in like_params]

    index = 0

    output = []

    for i in range(len(like_params)):
        output.append(flat_tensor_param[index:index+sizes[i]].reshape(like_params[i].size()))
        index += sizes[i]

    return output

def flatten_params(params):

    return t.cat([x.flatten() for x in params])

def cosine_similarity_params(params1, params2):

    norm1 = 0
    norm2 = 0

    dot_product = 0

    for p1, p2 in zip(params1, params2):
        dot_product += t.sum(p1*p2)
        norm1 += t.sum(p1**2)
        norm2 += t.sum(p2**2)

    return dot_product / (norm1**0.5 * norm2**0.5)



def find_closest_ray_in_vector_span(target_w, vector_basis, init_alpha=None, verbose=False):
    """
    target_w is a detached pytorch tensor
    vector_basis is a list of detached tensors of same size as target_w
    init_alpha should be a tensor of size len(vector_basis)
    """

    alpha = init_alpha if init_alpha is not None else t.ones(len(vector_basis))
    alpha.requires_grad = True

    # better this than SGD or Adam
    optimizer = optim.LBFGS([alpha],
                            history_size=10,
                            max_iter=20,
                            line_search_fn="strong_wolfe")

    # size [len(vector_basis), vector_basis[0].size(0)]
    vector_basis = t.stack(vector_basis, dim=0)

    target_w_norm = t.norm(target_w)

    # size [len(vector_basis)]
    dot_products = t.sum(target_w * vector_basis, dim=1)

    loss_item = 0

    # this is a function we provide to the LBFGS optimizer
    def closure():

        optimizer.zero_grad()

        # maximize cosine similarity between target_w and the spanned vector from vector_basis
        spanned_norm = t.norm(t.sum(alpha.unsqueeze(1) * vector_basis, dim=0))
        loss = -t.sum(alpha * dot_products) / (target_w_norm * spanned_norm)

        loss.backward()

        return loss

    for i in range(0, 1):

        # this is a function we provide to the LBFGS optimizer

        optimizer.step(closure)

        if verbose:
            loss_item = closure().item()
            print(f"{i}:{-loss_item}")

    if not verbose:
        loss_item = closure().item()

    alpha = (alpha/t.norm(alpha)).detach()

    return -loss_item, alpha, t.sum(alpha.unsqueeze(1) * vector_basis, dim=0).detach()


def get_video_from_env(env, actions, save_path=None, seed=None, alphas=None):
    """
    todo: better to sample actions from our agent
    Get a video from the environment using the given actions.

    :param env: The environment to get the video from.
    :param actions: The actions to use to get the video.
    :return: The video as a list of images.
    """
    video = []
    observation, info = env.reset(seed=seed)
    for i in range(len(actions)):
        action = actions[i]
        observation, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if alphas is not None:
            alpha = alphas[i]

            fraction_between_min_max = (alpha - np.min(alphas))/(np.max(alphas) - np.min(alphas))

            reds = int(256*(1-fraction_between_min_max)) * np.ones([frame.shape[0], 30, 1], dtype=np.uint8)
            greens = int(256*fraction_between_min_max) * np.ones([frame.shape[0], 30, 1], dtype=np.uint8)
            blues = np.zeros_like(reds, dtype=np.uint8)

            border = np.concatenate([reds, greens, blues], axis=2)
            frame = np.concatenate([frame, border], axis=1)

        video.append(frame)
        if terminated or truncated:
            break

    if save_path is not None:
        imageio.mimsave(save_path, video, fps=30)

    return video

def split_vectorized_traj(trajs):

    new_trajs = []

    for traj in trajs:

        obs = traj['obs']
        actions = traj['a']
        rew = traj['r']
        log_probs = traj['sampling_log_prob']

        n_env = obs.shape[1]

        for i in range(n_env):
            new_trajs.append({'obs': obs[:, i],
                              'a': actions[:, i],
                              'r': rew[:, i],
                              'sampling_log_prob': log_probs[i]})

    return new_trajs


def get_state_trajectory_from_action_seq(action_seq, env, seed=42):

    obs_list = []

    observation, info = env.reset(seed=seed)

    for i in range(0, len(action_seq)):
        new_obs, reward, terminated, truncated, info = env.step(action_seq[i])
        obs_list.append(observation)
        observation = new_obs

    return obs_list


def sample_trajectories(agent, env, n_traj=1000, max_steps=1000, random_sampling=False, explore_std=None, seed=None):
    """
    Sample trajectories from the environment using the model.

    :param model: The model to use to sample the trajectories.
    :param env: The environment to sample the trajectories from, assuming vectorized.
    :return: A list of trajectories. Each trajectory is a list of dictionaries with keys 'obs', 'a', 'r', 'sampling_log_prob'.

    traj['obs']: np_array of [max_steps, n_env, obs_shape]
    traj['a']: np_array of [max_steps, n_env, action_space_shape]
    traj['r']: np_array of [max_steps, n_env]
    traj['sampling_log_prob']: np_array of [n_env]
    """
    trajectories = []
    obs_traj = []
    a_traj = []
    r_traj = []

    n_iter = int(np.ceil(n_traj/env.num_envs))

    print(f'sampling {n_traj} trajs of length {max_steps}')

    for traj in range(n_iter):
        if traj%10 == 0: print(f"{traj}/{n_iter}")
        observation, info = env.reset(seed=seed)

        traj_log_prob = []
        for step in range(max_steps):

            # if step%10==0:
            #     print(f"traj{traj}/{n_iter} --- {step}/{max_steps}")

            if not random_sampling:
                action, action_log_prob = agent.sample(observation, veto_explore_std=explore_std)  # agent policy that uses the observation and info
            else:
                action = env.action_space.sample()
                # this is uniform sampling, we're treating all action spaces as between 0 and 1
                action_log_prob = np.zeros(env.num_envs)

            new_obs, reward, terminated, truncated, info = env.step(action)
            # reward = info['forward_reward'] if 'forward_reward' in info else reward  # set the reward to purely the forward component
            obs_traj.append(observation)
            a_traj.append(action)
            r_traj.append(reward)
            traj_log_prob.append(action_log_prob)

            observation = new_obs

            if terminated.any() or step == max_steps-1:
            # if step == max_steps - 1:
                if traj % 10 == 0:
                    print(f"terminated at step {step}")
                trajectories.append({'obs': np.stack(obs_traj, axis=0),
                                     'a': np.stack(a_traj, axis=0),
                                     'r': np.stack(r_traj, axis=0),
                                     'sampling_log_prob': sum(traj_log_prob)})


                # here trajectories[i]['obs'] will be [n_step, n_env, obs_space]

                # print(obs_traj[:10])
                obs_traj = []
                a_traj = []
                r_traj = []
                break

    return split_vectorized_traj(trajectories)

if __name__ == "__main__":

    from agents.PolicyGradientAgent import *
    import gymnasium as gym

    env_name = "HalfCheetah-v4"
    num_envs = 32
    env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=True)
    agent = PolicyGradientAgent(env, device=t.device('cpu'), exploration_std=0.1)

    trajs = sample_trajectories(agent, env, 20, max_steps=100, random_sampling=False)

    # obs, info = env.reset()
    # action, action_log_prob = agent.sample(obs)

    target_w = t.randn(6000)
    vector_basis = [t.randn(6000) for x in range(10)]

    for i in range(50):
        loss, alphas, _ = find_closest_ray_in_vector_span(target_w, vector_basis, verbose=True)
        vector_basis += [t.randn(6000) for x in range(10)]
        print(f"i:{i}, loss:{loss:.5f}")

