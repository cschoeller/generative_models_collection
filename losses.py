"""
Implements loss functions to train the generative models.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def vae_loss(batch, model):
    """ VAE likelihood lower-bound (ELBO), assuming gaussian latent distributions. """ 
    batch_size_norm = 1./batch.size(0)
    y_pred, mu, logvar = model(batch)
    reconstruction_error = ((batch - y_pred)**2).sum(1)
    KLD = 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    # Because we are in low-dimensional, low-value space, the contribution of the 
    # reconstruction error is minor and the KLD dominates, so we manually weight it down
    # to allow a more flexible embedding. For the GMM dataset this value should be
    # higher, e.g., 0.7.
    loss = batch_size_norm * (reconstruction_error - 0.1*KLD).sum()
    return loss, y_pred


def log_likelihood(x, mu, std):
    """ Compute the univariate gaussian log likelihood of the datapoint x. """
    # manually derived from log of gaussian pdf
    # mu = torch.tensor(mu)
    # std = torch.tensor(std)
    # var = (std**2)
    # log_l = -((x - mu) ** 2) / (2 * var) - 0.5 * torch.log(2 * math.pi * var)
    # return log_l

    # native pytorch gaussian
    normal = Normal(mu, std, validate_args=True)
    log_l = normal.log_prob(x)
    return log_l

def made_loss(batch, model):
    """
    The MADE model predicts gaussian parameters of the shape [mean_x, mean_y, std_x, std_y].
    Then we train with maximum likelihood by minimizing the negative log likelihood of the
    input data under the predicted gaussian. Because we compute the log of the likelihood we can
    just sum the individual conditional likelihoods that MADE models.
    """
    mu, std = model(batch)
    log_l = log_likelihood(batch, mu, std)

    # compute actual loss
    batch_size = log_l.size(0)
    nll_loss = (-1 * torch.sum(log_l)) / batch_size
    return nll_loss, log_l

def maf_loss(batch, model):
    """
    The likelihood of each datapoint can be computd with p(x) = p(z) * |det J_f^-1(x)|.
    As we want to minimize the NLL and in the MAF case the jacobian determinant is always positive,
    we can just compute log(p(x)) = log(p(z)) + log(det J_f^-1(x)).
    """
    z, jacobian_det = model(batch) # for MAF forward is f^-1(x)
    normal = Normal(0, 1, validate_args=True) # standard normal
    # note that the product of two univariate normal densities p(x1) * p(x2) results in a multivariate
    # density p(x1, x2) with diagonal covariance. In log space this turns into a sum. Instead, also a
    # standard multivariate normal could be used for evaluatuion of p(z). Same holds for the iaf_loss.
    log_l = normal.log_prob(z).sum(1) + torch.log(jacobian_det)

    # compute actual loss
    batch_size = log_l.size(0)
    nll_loss = (-1 * torch.sum(log_l)) / batch_size
    return nll_loss, log_l

def iaf_loss(batch, model):
    """
    The loss for the IAF is mathematically equivalent to that of the MAF, but we must use the
    backwards() function with the inverted transformation.
    """
    z, jacobian_det = model.backwards(batch) # for IAF backwards is f^-1(x)
    normal = Normal(0, 1, validate_args=True) # standard normal
    log_l = normal.log_prob(z).sum(1) + torch.log(jacobian_det)

    # compute actual loss
    batch_size = log_l.size(0)
    nll_loss = (-1 * torch.sum(log_l)) / batch_size
    return nll_loss, log_l

def nsf_loss(batch, model):
    z, log_abs_jacobian_det = model.inverse(batch)
    normal = Normal(0, 1, validate_args=True)
    log_l = normal.log_prob(z).sum(1) + log_abs_jacobian_det

    # compute actual loss
    batch_size = log_l.size(0)
    nll_loss = (-1 * torch.sum(log_l)) / batch_size
    return nll_loss, log_l

def nll(batch, model):
    """ Negative log likelihood loss. """
    log_prob = -1*model.log_prob(batch).mean(0)
    return log_prob, log_prob