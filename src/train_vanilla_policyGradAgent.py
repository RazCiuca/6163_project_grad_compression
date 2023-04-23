

# import sys
# sys.path.append('/')

# Now you can import modules from the parent directory
from Trainer import Trainer

if __name__ == "__main__":

    # for a list of environments

    # for a list of network architectures

    # for a list of random initializations

    # instantiate a Trainer and train the network, storing all intermediate states

    # also storing intermitent videos from our policies

    trainer = Trainer("HalfCheetah-v4", num_envs=320)

    for j in range(1000):
        average_reward_in_buffer = trainer.train(n_samples=500, n_iterations=1000,
                                  traj_batch_size=128, max_steps=300,
                                  verbose=True, sample_randomly_instead_of_agent=True)

        # agent = trainer.agent
        # env = trainer.env
        # buffer = trainer.buffer

        print(f"iteration {j}, average_reward:{average_reward_in_buffer}")

