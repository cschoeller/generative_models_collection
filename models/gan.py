"""
Implements the following generative model:

"Generative Adversarial Networks", Goodfellow et al., Jun 2014 (GAN)
https://arxiv.org/abs/1406.2661
"""


import torch
import torch.nn as nn


class GANDiscriminator(nn.Module):
    """
    The discriminator gets an input and predicts a probability of whether it is
    from the target distribution or not.
    """

    def __init__(self, nin=2, nh=[24, 24, 24], device=0):
        super().__init__()
        self.layers = [nin] + nh + [1] # binary output
        self.net = []
        for (l1, l2) in zip(self.layers, self.layers[1:]):
            self.net.extend([nn.Linear(l1, l2), nn.LeakyReLU()])
        self.net.pop() # remove last activation
        self.net.append(nn.Sigmoid()) # for binary classification
        self.net = nn.Sequential(*self.net)

        self.device = device
        #self.cuda(self.device)
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

class GANGenerator(nn.Module):
    """
    The generator takes random noise vectors and outputs samples from the
    target distribution.
    """
    def __init__(self, noise_size=8, nh=[24, 24, 24], nout=2, device=0):
        super().__init__()
        self.noise_dims = noise_size
        self.layers = [noise_size] + nh + [nout]
        self.net = []
        for (l1, l2) in zip(self.layers, self.layers[1:]):
            self.net.extend([nn.Linear(l1, l2), nn.LeakyReLU()])
        self.net.pop() # remove last relu
        self.net = nn.Sequential(*self.net)

        self.device = device
        #self.cuda(self.device)
        self.to(self.device)

    def forward(self, z):
        """ Receives a random vector and generates samples from p(x). """
        return self.net(z)
    
    def sample(self, n):
        z = torch.randn(n, self.noise_dims).to(self.device)
        return self.forward(z)