"""
Implements the following generative model:

"MADE: Masked Autoencoder for Distribution Estimation", Germain et al., Feb 2013 (MADE)
https://arxiv.org/abs/1502.03509
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import seaborn as sns
import matplotlib.pyplot as plt

from .made_layer import MadeLayer


class GaussianMADE(MadeLayer):

    def __init__(self, nin, hidden_sizes, device=0):
        """
        Implements the MADE model with a Gaussian output. Like this the number of outputs is
        always twice the inputs (mu, std). The number of hidden units per layer for the MADE
        layer can be defined. The number of masks is restricted to 1 and the input ordering to
        natural.
        """
        super().__init__(nin, hidden_sizes, 2*nin, num_masks=1, natural_ordering=True)
        self.device = device
        self.cuda(self.device)
    
    def forward(self, x):
        """
        Forward pass computes the parameters of the conditional densities.
        """
        y = super().forward(x)
        mu = y[:,:int(y.size(-1)//2)]
        alpha = y[:,int(y.size(-1)//2):] # predicted in log space
        std = torch.exp(alpha) # convert to std, ensures ensure positive values
        return (mu, std)

    def sample(self, n):
        """
        Sample from the learned distribution by drawing samples from each individual conditional and
        then re-run the network to parameterize the next conditional.
        NOTE: This implementation assumes natural ordering and a single mask!
        """
        assert(self.num_masks == 1)
        assert(self.natural_ordering == True)
        with torch.no_grad():
            samples = torch.Tensor([[0.0, 0.0]]).repeat(n, 1).to(self.device) # start with sample dummies
            # consecutively sample values for each of the n dimensions from the conditionals
            for i in range(samples.size(1)):
                mu, std = self(samples)
                normal = Normal(mu[:,i], std[:,i], validate_args=True)
                samples_i = normal.sample(torch.Size([1])).squeeze()
                samples[:,i] = samples_i
        return samples