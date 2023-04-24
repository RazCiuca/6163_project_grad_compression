

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

    ant_dict = {'env_name': env_names[1],
                'num_envs': 32,
                'lr' : 1e-4,
                'init_exploration_std': 0.1,
                'n_samples': 128,
                'n_iterations': 100,
                'max_IS': 1e4,
                'traj_batch_size': 128,
                'max_steps': 200}

    cheetah_dict = {'env_name': env_names[2],
                    'num_envs': 32,
                    'lr': 1e-4,
                    'init_exploration_std': 0.1,
                    'n_samples': 256,
                    'n_iterations': 100,
                    'max_IS': 1e4,
                    'traj_batch_size': 256,
                    'max_steps': 200}

    hopper_dict = {'env_name': env_names[3],
                   'num_envs': 1,
                   'lr': 1e-4,
                   'init_exploration_std': 0.1,
                   'n_samples': 256,
                   'n_iterations': 100,
                   'max_IS': 1e4,
                   'traj_batch_size': 256,
                   'max_steps': 200}

    pusher_dict = {'env_name': env_names[7],
                   'num_envs': 32,
                   'lr': 1e-4,
                    'init_exploration_std': 0.2,
                    'n_samples': 1024,
                    'n_iterations': 100,
                    'max_IS': 1e4,
                    'traj_batch_size': 1024,
                    'max_steps': 200}

    for type_model in ['quad', 'cube', 'net']:
        for exp_dict in [ant_dict, cheetah_dict, hopper_dict, pusher_dict]:
            for i in range(3):

                # experiment name to save with
                exp_name = type_model + '_' + exp_dict['env_name'] + '_' + str(i)

                os.mkdir('../data/' + exp_name)
                videos_folder = '../data/' + exp_name + '/videos'
                os.mkdir(videos_folder)

                trainer = Trainer(exp_dict, exp_dict['env_name'], exp_name,
                                  type_model, num_envs=exp_dict['num_envs'],
                                  init_exploration_std=exp_dict['init_exploration_std'],
                                  device=t.device('cpu'), lr=exp_dict['lr'])

                try:
                    for j in range(200):
                        average_reward_in_buffer = trainer.train(n_samples=exp_dict['n_samples'],
                                                                 n_iterations=exp_dict['n_iterations'],
                                                                 max_IS=exp_dict['max_IS'],
                                                                 traj_batch_size=exp_dict['traj_batch_size'],
                                                                 max_steps=exp_dict['max_steps'],
                                                                 verbose=True,
                                                                 sample_randomly_instead_of_agent=False,
                                                                 save_video_name= videos_folder + '/' + str(j) if j % 5 == 0 else None)

                        trainer.save_state()
                except:
                    continue

