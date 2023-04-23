

# import sys
# sys.path.append('/')
import torch as t
from utils import *
# Now you can import modules from the parent directory
from Trainer import Trainer
from torch.distributions import Beta, Normal


if __name__ == "__main__":

    # for a list of environments

    # for a list of network architectures

    # for a list of random initializations

    # instantiate a Trainer and train the network, storing all intermediate states

    # also storing intermitent videos from our policies

    trainer = Trainer("Ant-v4", num_envs=64, device=t.device('cpu'), init_exploration_std=0.1)

    # update the state means and variance with randomly sampled actions
    trainer.get_initial_state_mean_variance()

    for j in range(1000):
        average_reward_in_buffer = trainer.train(n_samples=1000, n_iterations=50,
                                              traj_batch_size=1024, max_steps=200,
                                              verbose=True, sample_randomly_instead_of_agent=False,
                                              save_video_name=str(j))

        # trainer.agent.explore_std

        # agent = trainer.agent
        # env = trainer.env
        # buffer = trainer.buffer
        # trajs = sample_trajectories(agent, env, n_traj=2, max_steps=3, random_sampling=False)
        #
        # print(trajs[0]['obs'])
        # # print(trajs[1]['sampling_log_prob'])
        #
        # buffer.buffer = []  # todo: remove once we know IS works
        # # add to buffer
        # buffer.add_trajectories(trajs)
        #
        # average_return = buffer.get_average_return()
        # agent.update_average_return(average_return)
        #
        # obs, actions, mean_rewards, log_probs, sample_indices = buffer.sample(2)
        #
        # trajs_batch = (obs, actions, mean_rewards, log_probs, sample_indices)
        # IS_ratios = agent.gradient_step(trajs_batch, print_grad_norm=True)


    # print(f"iteration {j}, average_reward:{average_reward_in_buffer}")

