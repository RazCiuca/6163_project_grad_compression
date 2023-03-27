
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
        self.nodes = nodes
        self.h_nodes = nodes[1:-1]  # all the hidden nodes
        # initialize the weights for all layers, with std = 1/nodes[i]**0.5
        self.weights = nn.ParameterList([nn.Parameter(t.randn(nodes[i], nodes[i+1])/(nodes[i]**0.5), requires_grad=True)
                        for i in range(0, len(nodes)-1)])
        # initialize the biases to 0
        self.biases = nn.ParameterList([nn.Parameter(t.zeros(nodes[i+1]), requires_grad=True)
                       for i in range(0, len(nodes)-1)])
        # list containing our transition functions, all ReLU instead of the last one, which is just the identity
        self.sigmas = [F.relu for _ in range(0, len(self.weights)-1)] + [lambda x: x]

    def forward(self, inputs):
        """
        :param inputs: assumed to be of size [batch_size, self.nodes[0]]
        :return: returns the output tensor, of size [batch_size, self.nodes[-1]]
        """

        x = inputs

        for w, b, sigma in zip(self.weights, self.biases, self.sigmas):
            x = sigma(x @ w + b)

        return x

    def scale_weights_with_symmetry(self, scales):
        """
        :param scales: list of Tensors of length len(self.h_nodes), each tensor has shape [self.h_nodes[i]]
        :return: None
        """
        for i in range(0, len(self.h_nodes)):

            # scale the in-weights and their gradients
            self.weights[i].data = self.weights[i].data * scales[i].unsqueeze(0)
            self.biases[i].data = self.biases[i].data * scales[i]

            self.weights[i].grad = self.weights[i].grad / scales[i].unsqueeze(0)
            self.biases[i].grad = self.biases[i].grad / scales[i]

            # scale the out-weights and their gradients
            self.weights[i+1].data = self.weights[i+1].data / scales[i].unsqueeze(1)

            self.weights[i+1].grad = self.weights[i+1].grad * scales[i].unsqueeze(1)

    def get_grad_L2_norm(self):
        """
        :return: the L2 norm of the network parameters
        """
        return sum([w.grad.norm()**2 + b.grad.norm()**2 for w, b in zip(self.weights, self.biases)])**0.5

    def get_grad_sq_L2_norms(self):
        """
        :return: returns two lists of tensors of shape [self.h_nodes[i]] corresponding to the in-weight gradient squared
                 norms and the out-weight gradient squared norms. We include the bias terms in the in-weight norms
        """
        squared_w_grads = [w.grad**2 for w in self.weights]

        in_weight_sq_norms = [t.sum(w2, dim=0) + b.grad**2 for w2, b in zip(squared_w_grads[:-1], self.biases[:-1])]
        out_weights_sq_norms = [t.sum(w2, dim=1) for w2 in squared_w_grads[1:]]

        return in_weight_sq_norms, out_weights_sq_norms

    def get_optimal_scales(self, eps_norm=1e-12):
        """
        :param eps_norm: threshold for considering a node to have zero gradiens
        :return: a list of tensors corresponding to the optimal scales to minimize the gradient norm (locally)
                 and the gradient norm for the derivatives
        """
        # these are the squared L2 norms of the gradients for the in-weights of each node and the out-weights
        in_weight_sq_norms, out_weights_sq_norms = self.get_grad_sq_L2_norms()
        # the optimal scaling factor is easy to compute from the gradient norms, see function description

        # if the either the out, or in-norms are close zero, don't do a scaling factor for that node
        scale_masks = [t.logical_and(out_norms > eps_norm, in_norms > eps_norm)
                       for in_norms, out_norms in zip(in_weight_sq_norms, out_weights_sq_norms)]

        scales = [(in_norms / (eps_norm + out_norms)) ** (1 / 4) * mask + t.logical_not(mask).float()
                  for in_norms, out_norms, mask in
                  zip(in_weight_sq_norms, out_weights_sq_norms, scale_masks)]

        gradient_norm_for_a = sum(t.sum((in_norms - out_norms) ** 2)
                                  for in_norms, out_norms in zip(in_weight_sq_norms, out_weights_sq_norms)) ** 0.5

        return scales, gradient_norm_for_a

    def minimize_grad_norm_along_scaling_sym(self, gamma=None, n_iter=1, eps=1e-3, verbose=False):
        """
        The total contribution of a node to the squared L2 norm of the gradients will be the following:

        in_w_grad_sq_norm / a**2 + out_w_grad_sq_norm * a**2

        where a>0 is a symmetry parameter that we need to set. This is of the form X/a**2 + Y*a**2 = Z, we get
        dZ/da = -2X/a**3 + 2Y*a , if we set this to zero we get

        X = Y * a**4  ==>  a = (X/Y)**(1/4)

        Or we can use a learning rate to set a = 1 - lr * (-2X/a**3 + 2Y*a)
        where we've used that a_initial = 1 for each iteration of the algorithm

        :param gamma: between 0 and 1, if given, only one iteration is done, and the scales are put to this power
        :param eps: desired final
        :return: final gradient norm
        """

        if gamma is not None:
            for i in range(n_iter):
                scales, _ = self.get_optimal_scales()
                self.scale_weights_with_symmetry([scale**(gamma/n_iter) for scale in scales])
        else:
            i = 0
            while i < 20:
                if verbose:
                    print(f"at iter {i} of minimization, grad norm is: {self.get_grad_L2_norm()}")

                scales, gradient_norm_for_a = self.get_optimal_scales()

                # finally rescale the nodes
                self.scale_weights_with_symmetry(scales)

                i += 1
                if gradient_norm_for_a < eps:
                    break

        return self.get_grad_L2_norm()

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


if __name__ == "__main__":

    # this is just a dummy test to make sure everything works fine

    batch_size = 64
    model = MLP([20, 20, 20, 20])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    inputs = t.randn(batch_size, 20)
    outputs = model(inputs)
    objective = outputs.mean()
    objective.backward()

    # make sure scaling params doesn't affect outputs:
    model.scale_weights_with_symmetry([t.rand(x) for x in model.h_nodes])
    scaled_grads = [x.grad.data.clone() for x in model.parameters()]
    outputs_2 = model(inputs)
    print(f"standard deviation of output differences:  {(outputs-outputs_2).std()}")

    # and make sure that scaling correctly computes
    optimizer.zero_grad()
    outputs_2.mean().backward()
    new_grads = [x.grad.data for x in model.parameters()]

    mean_squared_loss = sum([t.sum((x-y)**2) for x,y in zip(new_grads, scaled_grads)])
    print(f"L2 norm of difference between true gradient and scaled gradient: {mean_squared_loss**0.5}")

    model.minimize_grad_norm_along_scaling_sym()
