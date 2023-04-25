

import os
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

    env_names = {1: 'Ant-v4',
                 2: 'HalfCheetah-v4',
                 3: 'Hopper-v4',
                 4: 'HumanoidStandup-v4',
                 5: 'Humanoid-v4',
                 6: 'Reacher-v4',
                 7: 'Pusher-v4',
                 8: 'Walker2d-v4'}

    # DONT CHANGE, this works
    ant_dict = {'env_name': env_names[1],
                'num_envs': 16,
                'lr' : 1e-4,
                'init_exploration_std': 0.1,
                'n_samples': 1024,
                'n_iterations': 50,
                'max_IS': 4e4,
                'traj_batch_size': 1024,
                'max_steps': 200}

    # DONT CHANGE, this works
    cheetah_dict = {'env_name': env_names[2],
                    'num_envs': 8,
                    'lr': 1e-3,
                    'init_exploration_std': 0.1,
                    'n_samples': 64,
                    'n_iterations': 50,
                    'max_IS': 4e4,
                    'traj_batch_size': 64,
                    'max_steps': 1000}

    # doesn't work, it mostly just stays upright to get the healthy reward proportional to episode length
    walked2d_dict = {'env_name': env_names[8],
                   'num_envs': 8,
                   'lr': 1e-4,
                   'init_exploration_std': 0.1,
                   'n_samples': 256,
                   'n_iterations': 50,
                   'max_IS': 4e4,
                   'traj_batch_size': 256,
                   'max_steps': 500}

    # doesn't work, it mostly just stays upright to get the healthy reward proportional to episode length
    hopper_dict = {'env_name': env_names[3],
                   'num_envs': 4,
                   'lr': 1e-4,
                   'init_exploration_std': 0.1,
                   'n_samples': 256,
                   'n_iterations': 50,
                   'max_IS': 4e4,
                   'traj_batch_size': 256,
                   'max_steps': 1000}

    # DONT CHANGE, this works
    pusher_dict = {'env_name': env_names[7],
                   'num_envs': 16,
                   'lr': 1e-5,
                    'init_exploration_std': 0.2,
                    'n_samples': 256,
                    'n_iterations': 100,
                    'max_IS': 1e4,
                    'traj_batch_size': 256,
                    'max_steps': 200}

    humanoid_dict = {'env_name': env_names[5],
                     'num_envs': 1,
                     'lr': 1e-7,
                     'init_exploration_std': 0.1,
                     'n_samples': 64,
                     'n_iterations': 50,
                     'max_IS': 1e3,
                     'traj_batch_size': 64,
                     'max_steps': 1000}

    # for type_model in ['quad', 'cubic', 'net']:
    for type_model in ['quad']:
        for exp_dict in [ant_dict, cheetah_dict, pusher_dict]:
            for i in [4]:

                # experiment name to save with
                exp_name = type_model + '_' + exp_dict['env_name'] + '_' + str(i)

                os.mkdir('../data/' + exp_name)
                videos_folder = '../data/' + exp_name + '/videos'
                os.mkdir(videos_folder)

                trainer = Trainer(exp_dict, exp_dict['env_name'], exp_name,
                                  type_model, num_envs=exp_dict['num_envs'],
                                  init_exploration_std=exp_dict['init_exploration_std'],
                                  device=t.device('cpu'), lr=exp_dict['lr'])

                # if not humanoid, find state means and variances
                if exp_dict['env_name'] != env_names[5]:
                    trainer.get_initial_state_mean_variance()

                try:
                    for j in range(200):
                        average_reward_in_buffer = trainer.train(n_samples=exp_dict['n_samples'],
                                                                 n_iterations=exp_dict['n_iterations'],
                                                                 max_IS=exp_dict['max_IS'],
                                                                 traj_batch_size=exp_dict['traj_batch_size'],
                                                                 max_steps=exp_dict['max_steps'],
                                                                 verbose=True,
                                                                 sample_randomly_instead_of_agent=False,
                                                                 save_video_name= videos_folder + '/' + str(j) if j % 1 == 0 else None)

                        trainer.save_state()
                except:
                    continue

