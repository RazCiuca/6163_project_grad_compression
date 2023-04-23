from MLP import *
import imageio

def get_video_from_env(env, actions, save_path=None):
    """
    todo: better to sample actions from our agent
    Get a video from the environment using the given actions.

    :param env: The environment to get the video from.
    :param actions: The actions to use to get the video.
    :return: The video as a list of images.
    """
    video = []
    observation, info = env.reset()
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        video.append(env.render(mode='rgb_array'))
        if terminated or truncated:
            break

    if save_path is not None:
        imageio.mimsave(save_path, video)

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

def sample_trajectories(agent, env, n_traj=1000, max_steps=1000, random_sampling=False):
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

    n_iter = int(n_traj/env.num_envs) + 1

    for traj in range(n_iter):

        obs, info = env.reset()

        traj_log_prob = 0
        for step in range(max_steps):

            # if step%10==0:
            #     print(f"traj{traj}/{n_iter} --- {step}/{max_steps}")

            if not random_sampling:
                action, action_log_prob = agent.sample(obs)  # agent policy that uses the observation and info
                action = action.numpy()
            else:
                action = env.action_space.sample()
                # this is uniform sampling, we're treating all action spaces as between 0 and 1
                action_log_prob = np.zeros(env.num_envs)

            observation, reward, terminated, truncated, info = env.step(action)
            obs_traj.append(observation)
            a_traj.append(action)
            r_traj.append(reward)
            traj_log_prob += action_log_prob

            if terminated.any() or truncated.any() or step == max_steps-1:
                trajectories.append({'obs': np.stack(obs_traj, axis=0),
                                     'a': np.stack(a_traj, axis=0),
                                     'r': np.stack(r_traj, axis=0),
                                     'sampling_log_prob': traj_log_prob})
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
    agent = PolicyGradientAgent(env)

    trajs = sample_trajectories(agent, env, 20, max_steps=100, random_sampling=False)

    # obs, info = env.reset()
    # action, action_log_prob = agent.sample(obs)
