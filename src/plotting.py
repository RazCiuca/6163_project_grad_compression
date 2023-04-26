

# - Plot of the cosine similarity as a function of traj_len for Ant with quad and net, both for using alphas and not using alphas
# - Plot the average reward for Ant with quad and net, along a line search in the direction pointed to by the compressed gradient
# - Plot the cosine similarity for Ant for quad/net with a random weight vector as a function of traj_len

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def get_cosine_sim_data(path):

    filenames = sorted(os.listdir(path))
    x_data = np.array([int(x.split('_')[1]) for x in filenames])
    sorted_indices = np.argsort(x_data)

    data = [pickle.load(open(path + file, 'rb')) for file in filenames]

    y_data = np.array([x['cosine_sim'] for x in data])

    return x_data[sorted_indices], y_data[sorted_indices]

if __name__ == "__main__":

    x_label = "Length of Trajectory"
    y_label = "Cosine Similarity"
    title = "Cosine Sim. vs. length of optimized trajectory on Ant-v4"
    filename = None

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    x_ant_quad_alpha, y_ant_quad_alpha = get_cosine_sim_data("../data/compressor_video_ant_quad/action_seqs/")
    x_ant_quad_noalpha, y_ant_quad_noalpha = get_cosine_sim_data("../data/compressor_video_ant_quad_noalpha/action_seqs/")
    x_ant_net_alpha, y_ant_net_alpha = get_cosine_sim_data("../data/compressor_video_ant_net/action_seqs/")
    x_ant_net_noalpha, y_ant_net_noalpha = get_cosine_sim_data("../data/compressor_video_ant_net_noalpha/action_seqs/")

    x_ant_quad_alpha_random_weight, y_ant_quad_alpha_random_weight = get_cosine_sim_data("../data/compressor_video_ant_quad_random_weight/action_seqs/")
    x_ant_net_alpha_random_weight, y_ant_net_alpha_random_weight = get_cosine_sim_data("../data/compressor_video_ant_net_random_weight/action_seqs/")

    # Plot the data
    plt.plot(x_ant_quad_alpha, y_ant_quad_alpha, linewidth=2, linestyle='solid', color='blue', label="quad model, alpha-optim")
    plt.plot(x_ant_quad_noalpha, y_ant_quad_noalpha, linewidth=2, linestyle='solid', color='red', label="quad model, no alpha-optim")
    plt.plot(x_ant_net_alpha, y_ant_net_alpha, linewidth=2,linestyle='dotted', color='blue', label="net model, alpha-optim")
    plt.plot(x_ant_net_noalpha, y_ant_net_noalpha, linewidth=2, linestyle='dotted', color='red', label="net model, no alpha-optim")

    plt.plot(x_ant_quad_alpha_random_weight, y_ant_quad_alpha_random_weight, linestyle='solid', linewidth=2, color='green', label="quad model, random direction")
    plt.plot(x_ant_net_alpha_random_weight, y_ant_net_alpha_random_weight, linestyle='dotted', linewidth=2, color='green', label="net model, random direction")

    # Customize the graph with grid, linewidth, and line color
    ax.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()

    # If a filename is provided, save the graph as an image
    if filename:
        plt.savefig(filename)

    # Show the graph
    plt.show()

