
import gymnasium as gym
from utils import *
from ReplayBuffer import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    num_envs = 64

    # cheetah has obs_space of size 17 and action_space of size 6
    # training_env = gym.vector.make('HalfCheetah-v4', num_envs=num_envs, asynchronous=True)
    # env = gym.make('HalfCheetah-v4')

    training_env = gym.vector.make('HalfCheetah-v4', num_envs=num_envs, asynchronous=True)
    training_env.reset(seed=42)
    obs = []
    total_reward = np.zeros(num_envs)

    for i in range(1000):
        if i%100 == 0:
            print(f"step {i}")
        action = training_env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = training_env.step(action)

        obs.append(observation)
        total_reward += reward

    # now an np array of shape 1000, 64, 17
    obs = np.stack(obs, axis=0)

    training_env.close()
