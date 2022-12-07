"""
Implements the MADE model as a flexible layer. A single MadeLayer is used to implement MADE itself,
but its also a basic block of MAF and IAF. This file is based on the implementation of Andrej Karpathy,
but modified.

Source: https://github.com/karpathy/pytorch-normalizing-flows

NOTE:
Ensembling could be improved, such that the mask updates are handeled internally. Furthermore, a function
to easily re-arrange the outputs into natural ordering would be useful for application.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import seaborn as sns
import matplotlib.pyplot as plt


class MaskedLinear(nn.Linear):
    """
    Same as Linear, except it has a configurable binary mask over the weights to
    enable an autoregessive connectivity.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MadeLayer(nn.Module):

    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        A single made layer consisting of masked linear hidden layers.
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds, i.e. the output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train anensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        super().__init__()

        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert(self.nout % self.nin == 0)
        
        # define a MLP of maskable hidden layers
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # remove output activations
        self.net = nn.Sequential(*self.net)
        
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial connectivity self.m
        
        # NOTE: We could also precompute the masks and cache them, but this
        # could become expensive in terms of memory.
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed (ensures that each MADE gets the same connectivty in case we call update_masks()
        # multiple times for ensembling).
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample order inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin) # input layer

        for l in range(L): # hidden layers
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:]) # output layer, same ordering as inputs

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x):
        """
        Compute the parameters of the conditional densities. If the number of output nodes is
        greater than the number of input nodes, it is ordered in the following way (gaussian example):
        [mu_1, mu_2, mu_3, std_1, std_2, std_3], where mu_i and std_i have the same conditional connectivity
        pattern. In this case the masks are created for the mu and then duplicated for the stds.
        """
        y = self.net(x)
        return y

class CMadeLayer(nn.Module):

    def __init__(self, nin, hidden_sizes, nout, ncond=0, num_masks=1, natural_ordering=False):
        """
        Conditional version of the MadeLayer. Accepts an additional input for conditioning.

        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        ncond: number of conditional inputs, are expected to be appeneded at the end of the input and
               connect to all output variables and each hidden unit.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.ncond = ncond # conditional inputs
        assert(self.nout % self.nin == 0)

        # define a simple MLP neural net
        self.net = []
        hs = [nin + ncond] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # remove output layer ReLU
        self.net = nn.Sequential(*self.net)
        
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed (ensures that each MADE gets the same connectivty in case we call update_masks()
        # multiple times for ensembling).
        rng = np.random.RandomState()
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin) # input layer

        for l in range(L): # hidden layers
            #NOTE: Can be dangerous in case it drops connections completely?
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:]) # output layer, same ordering as inputs

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # NOTE: Necessary for conditioning.
        # adapt mask of first hidden layer to connect each node to all conditional inputs.
        # This is achieved by manually concatenating a block of ones, i.e., a 'fully connected'
        # block, to the mask matrix
        cond_mask = np.ones((self.ncond, masks[0].shape[1]))
        masks[0] = np.concatenate((masks[0], cond_mask), axis=0)
        
        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x, c=None):
        """
        Compute the parameters of the conditional densities. If the number of output nodes is
        greater than the number of input nodes, it is ordered in the following way (gaussian example):
        [mu_1, mu_2, mu_3, std_1, std_2, std_3], where mu_i and std_i have the same conditional connectivity
        pattern. In this case the masks are created for the mu and then duplicated for the stds.
        """
        if c is not None: # concatenate on non-batch dimension
            assert(len(c.shape) == 2)
            x = torch.cat((x, c), 1)
        y = self.net(x)
        return y