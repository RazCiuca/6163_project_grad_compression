
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    This class defines an MLP with arbitrary layers and nodes
    """
    def __init__(self, nodes):
        """
        :param nodes: list containing integers corresponding to our numbers of nodes in each layer
        """
        super(MLP, self).__init__()

        self.s = 1

        self.nodes = nodes
        self.h_nodes = nodes[1:-1]  # all the hidden nodes
        # initialize the weights for all layers, with std = 1/nodes[i]**0.5
        self.weights = nn.ParameterList([nn.Parameter(self.s * t.randn(nodes[i], nodes[i+1])/(nodes[i]**0.5), requires_grad=True)
                        for i in range(0, len(nodes)-1)])
        # initialize the biases to 0
        self.biases = nn.ParameterList([nn.Parameter(t.zeros(nodes[i+1]), requires_grad=True)
                       for i in range(0, len(nodes)-1)])
        # list containing our transition functions, all ReLU instead of the last one, which is just the identity
        self.sigmas = [t.relu for _ in range(0, len(self.weights)-1)] + [lambda x: x]


    def forward(self, inputs):
        """
        :param inputs: assumed to be of size [batch_size, self.nodes[0]]
        :return: returns the output tensor, of size [batch_size, self.nodes[-1]]
        """

        x = inputs

        for w, b, sigma in zip(self.weights, self.biases, self.sigmas):
            x = sigma(x @ w + b)

        return x

    def set_output_std(self, inputs, out_std):
        """
        changes the weights of the network uniformly so that the output standard deviation is set to the desired level
        :param inputs: tensor of inputs with a large batch size
        :param out_std: wanted output standard deviation
        :return: None
        """
        current_std = (self.forward(inputs)).std().item()

        # if the output variance is just zero, scale it up until we get the correct answer
        while current_std < out_std:
            scaling_factor = 100**(1.0/len(self.weights))
            for w, b in zip(self.weights, self.biases):
                w.data = w.data * scaling_factor
                b.data = b.data * scaling_factor

            current_std = (self.forward(inputs)).std().item()
            print(f"current std now: {current_std:.5e}")

        scaling_factor = (out_std/current_std)**(1.0/len(self.weights))

        for w, b in zip(self.weights, self.biases):
            w.data = w.data * scaling_factor
            b.data = b.data * scaling_factor

class MultidimensionalQuadraticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, order=2):
        super(MultidimensionalQuadraticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.order = order

        # Linear layer for quadratic coefficients

        self.second_order_dim = input_dim * input_dim
        self.third_order_dim = input_dim * input_dim * input_dim if order == 3 else 0

        self.w = nn.Parameter(t.zeros(self.third_order_dim + self.second_order_dim + input_dim, output_dim), requires_grad=True)
        self.b = nn.Parameter(t.zeros(output_dim), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        x_second_order = t.einsum('bi, bj -> bij', x, x).reshape(batch_size, self.second_order_dim)

        inputs = [x, x_second_order]

        if self.order == 3:
            x_third_order = t.einsum('bi, bj, bk -> bijk', x, x, x).reshape(batch_size, self.third_order_dim)
            inputs.append(x_third_order)

        # Concatenate interaction terms with the original features
        x_extended = t.cat(inputs, dim=1)

        # Apply linear layer to learn the coefficients
        output = x_extended @ self.w + self.b

        return output

if __name__ == "__main__":

    # this is just a dummy test to make sure everything works fine

    device = t.device('cuda')

    batch_size = 64
    model = MultidimensionalQuadraticRegression(20, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    inputs = t.randn(batch_size, 20).to(device)
    outputs = model(inputs)
    objective = outputs.mean()
    objective.backward()

