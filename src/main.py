"""
sample script to train reinforce with a MLP policy

"""

from utils import *
from ReplayBuffer import *
from src.agents.agent import RLagent

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # num_envs = 64

    # cheetah has obs_space of size 17 and action_space of size 6
    # training_env = gym.vector.make('HalfCheetah-v4', num_envs=num_envs, asynchronous=True)
    env = gym.make('HalfCheetah-v4')
    training_iters = 100
    agent = RLagent(obs_shape=17, action_shape=6)
    replay_buffer = ReplayBuffer()

    env_interact_batch_size = 1000
    optim_batch_size = 500

    optim_iters = 50

    for i in range(0, training_iters):

        # ==================
        # sample from environment and add to replay buffer
        # ==================

        trajectories = sample_trajectories(agent, env, n_traj=env_interact_batch_size, max_steps=500)
        replay_buffer.add_trajectories(trajectories)

        # print mean return of trajectories

        # ==================
        # take training steps with agent.train()
        # ==================
        agent.train(replay_buffer, optim_iters, optim_batch_size)

        # save model once in a while

        # save one of the trajectory videos once a while

        # plot stuff

    env.close()
