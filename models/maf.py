"""
Implements the following normalizing flows:

"Masked Autoregressive Flow for Density Estimation", Papamakarios et al., May 2017 (MAF)
https://arxiv.org/abs/1705.07057

"Improved Variational Inference with Inverse Autoregressive Flow", Kingma et al., June 2016 (IAF)
https://arxiv.org/abs/1606.04934

TODO: Convert jacobian determinants in log space.
"""


import torch
import torch.nn as nn

from .made_layer import *
from .invertible_batch_norm import InvertibleBatchNorm


class MAF(nn.Module):
    """
    Masked Autoregressive Flow that uses MADE layers for the linear output transformation.

    It models the following transforms:
    f(x;x) -> z     [cheap; one network pass for density evaluation]
    f^-1(z;x) -> x  [expensive; requires multiple network passes due to autoregression]
    """

    def __init__(self, nin, num_mades=5, nh=24, device=0, batchnorm=False):
        super().__init__()
        self.nin = nin
        self.batchnorm = batchnorm

        # initialize MADE layers
        self.flow_modules = torch.nn.ModuleList()
        for i in range(num_mades):

            if self.batchnorm and i > 0: # add batchnorm after intermediate layers
                bn = InvertibleBatchNorm(nin)
                self.flow_modules.append(bn)

            made = MadeLayer(nin, [nh, nh, nh], 2*nin, num_masks=1, natural_ordering=True)
            made.update_masks() # initialize one connectivity mask per made
            self.flow_modules.append(made)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        """
        The forward pass in MAF evaluates f(x;x) -> z. This allows to easily compute p(x) in terms of
        p(z) and the corresponding jacobian determinant.
        """
        jacobian_det = 1. # will be multiplied by all successive jacobian determinants
        for i, layer in enumerate(self.flow_modules):

            if type(layer) == MadeLayer: # made
                y = layer(x)
                mu, alpha = y[:,:int(y.size(-1)//2)], y[:,int(y.size(-1)//2):]
                x = self.transform_inv(x, alpha, mu) # this becomes z
                jacobian_det *= torch.exp((-1 * alpha).sum(1))
            else: #  batchnorm
                x, bn_jacobian_det  = layer(x)
                jacobian_det *= bn_jacobian_det

            # reverse output dimensions of every second MADE module to not condition the first
            # dimension on a single univariate gaussian as in a single MADE
            x = x if i%2 == 0 else x.flip(dims=(1,))

        return x, jacobian_det
    
    def backwards(self, z):
        """
        Takes z and computes the inverse transform to get corresponding x. To do this, we have to
        transform back from the last MADE, to the previous, until we reach x.
        As the transform in each MADE is parameterized by its previous inputs of the forward pass, we
        must recursively transform one dimension at a time, i.e. x_i = f^-1(z_i;x<i). This is why sampling
        is expensive in MAF.
        """
        jacobian_det = 1.
        for j, layer in reversed(list(enumerate(self.flow_modules))): # transform back from made_n to made_0
            z = z if j%2 == 0 else z.flip(dims=(1,)) # if the output of the made has been reversed, do it again

            if type(layer) == MadeLayer: # made
                made_jacobian_det = None
                for i in range(z.size(1)): # invert made from z_n -> z_n-1
                    y = layer(z.clone()) # clone to avoid in-place op errors
                    mu, alpha = y[:,:int(y.size(-1)//2)], y[:,int(y.size(-1)//2):]
                    z[:, i] = self.transform(z[:, i], alpha[:,i], mu[:,i]) # this becomes x
                    made_jacobian_det = alpha[:,i] if made_jacobian_det == None else (made_jacobian_det + alpha[:,i])
                made_jacobian_det = torch.exp(-1*made_jacobian_det)
                jacobian_det *= made_jacobian_det
            else: # batchnorm
                z, new_jacobian_det  = layer.backwards(z)
                jacobian_det *= new_jacobian_det

        return z, jacobian_det

    def transform(self, z, alpha, mu):
        return z * torch.exp(alpha) + mu

    def transform_inv(self, x, alpha, mu):
        return (x - mu) / torch.exp(alpha)

    def sample(self, n):
        """ Sample z from a standard normal and map them back to x with the inverse transform. """
        with torch.no_grad():
            samples = torch.zeros([n, self.nin]).to(self.device)
            z = torch.randn_like(samples)
            x, jacobian =  self.backwards(z)
            return x


class IAF(MAF):
    """
    Inverse Autoregressive Flow that is constructed like MAF, but reverses forward and 
    inverse operation, favoring efficient sampling.
    """

    def __init__(self, nin, num_mades=5, nh=24, device=0):
        super().__init__(nin, num_mades, nh, device, batchnorm=False)

    def forward(self, z):
        return super().backwards(z)

    def backwards(self, x):
        return super().forward(x)

    def sample(self, n):
        with torch.no_grad():
            samples = torch.zeros([n, self.nin]).to(self.device)
            z = torch.randn_like(samples)
            x, jacobian = self.forward(z)
            return x


class CIAF(nn.Module):
    """
    Inverse Autoregressive Flow that uses made layers as conditioners for the linear
    output transformation. This is an explicitly implemented conditional version of the
    same model.
    """

    def __init__(self, nin, num_mades=5, nh=[24, 24, 24], ncond=0, device=0, batchnorm=False):
        super().__init__()
        self.nin = nin # number of inputs to be transformed
        self.ncond = ncond # number of conditional inputs
        self.batchnorm = batchnorm

        if any(h < nin*2 for h in nh):
            raise ValueError("Number of hidden layers should be at least double of input size."
                             + " Important to maintain proper connectivity.")

        # initialize made layers
        self.flow_modules = torch.nn.ModuleList()
        for i in range(num_mades):

            if self.batchnorm and i > 0: # add batch after intermediate layers
                bn = InvertibleBatchNorm(nin)
                self.flow_modules.append(bn)

            made = CMadeLayer(nin, nh, 2*nin, ncond=ncond, num_masks=1, natural_ordering=True)
            made.update_masks() # initialize one connectivity mask per made
            self.flow_modules.append(made)

        self.device = device
        self.cuda(self.device)

    def forward(self, z, c=None):
        """
        The forward pass in IAF evaluates f(z;z) -> x. This allows to easily compute sample
        from p(x) by sampling from p(z).

        NOTE: In forward, as well as backwards its also possible to directly compute the
        log jacobian determinant, then the determinant computation just becomes a sum.
        """
        # jacobian_det = 1. # will be multiplied by all successive jacobian determinants
        log_jacobian_det = 0.
        for i, made in enumerate(self.flow_modules): # transform through all mades
            y = made(z, c)
            mu, alpha = y[:,:int(y.size(-1)//2)], y[:,int(y.size(-1)//2):]
            z = self.transform(z, alpha, mu)
            log_jacobian_det += alpha.sum(1) # not needed at the moment

            # reverse output dimensions of every second made module to not condition the first
            # dimension on a single univariate gaussian as in a single MADE
            z = z if i%2 == 0 else z.flip(dims=(1,)) # permutation is volume preserving (jacobian det 1)

        return z, log_jacobian_det

    def backwards(self, x, c=None):
        """
        Takes x and computes the inverse transform to get corresponding z. To do this, we have to
        transform back from the last MADE, to the previous, until we reach z.
        As the transform in each MADE is parameterized by its input, this must recursively transform
        one dimension at a time, i.e. z_i = f^-1(x_i;z<i). This is why density evaluation is expensive
        in IAF.
        """
        jacobian_det = 1.
        for j, made in reversed(list(enumerate(self.flow_modules))): # transform back from MADE_n to MADE_1
            x = x if j%2 == 0 else x.flip(dims=(1,)) # if the output of the made has been reversed, do it again
            made_jacobian_det = 0 #None
            for i in range(x.size(1)): # invert made from z_n -> z_n-1
                y = made(x.clone(), c) # clone to avoid in-place op errors
                mu, alpha = y[:,:int(y.size(-1)//2)], y[:,int(y.size(-1)//2):]
                x[:, i] = self.transform_inv(x[:, i], alpha[:,i], mu[:,i]) # this becomes z

                # jacobian det of each MADE must be constructed recursively as well, because alpha_i, of which
                # the jacobian consists, is only properly parameterized once z_(i-1) is fed into the network
            made_jacobian_det = torch.exp((-1 * alpha).sum(1)) # all alpha_i are fully updated now
            jacobian_det *= made_jacobian_det

        return x, jacobian_det

    def transform(self, z, alpha, mu):
        return z * torch.exp(alpha) + mu

    def transform_inv(self, x, alpha, mu):
        return (x - mu) * torch.exp(-1*alpha)

    def sample(self, n, c=None):
        """ Sample z from a standard normal and map them back to x with the transform. """
        with torch.no_grad():
            samples = torch.zeros([n, self.nin]).to(self.device)
            z = torch.randn_like(samples)
            x, jacobian = self.forward(z, c)
            return x
