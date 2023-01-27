"""
Implements a denoising diffusion model based on the papers:

"Denoising Diffusion Probabilistic Models", Ho et al., Dec 2020
"Understanding Diffusion Models: A Unified Perspective", Calvin Luo, Aug 2022
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.datasets import FakeData
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt


def forward_diffusion(x_0, alpha_bar, timesteps):
    """ Models function q(x_t | x_0) """
    x_0 /= 3. # normalize in range [-1,1]
    eps_0 = torch.randn_like(x_0).to(x_0.device)
    alpha_t_bar = alpha_bar[timesteps, None]
    x_t = torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1 - alpha_t_bar) * eps_0
    return eps_0, x_t

def create_alphas(T):
    """ Create alpha schedule """
    beta = .3 * torch.linspace(0., 0.85, T)**3 # polynomial
    alpha = 1 - beta
    return alpha

def diffusion_loss(x, model):
    """ MSE loss between sampled noise and predicted noise """
    T, alpha_bar = model.T, model.alpha_bar
    timestep = torch.randint(T, size=(x.shape[0],)).to(x.device) # sample timesteps
    
    # predict noise and compute loss
    eps_0, x_t = forward_diffusion(x, alpha_bar, timestep) # generate noisy inputs
    eps_pred = model(x_t, timestep)
    loss = ((eps_0 - eps_pred)**2).mean() # simplified loss, 'Ho et al. 2020'
    return loss, eps_pred

class DiffusionModel(nn.Module):

    def __init__(self, dim, T, device=0):
        super().__init__()
        self.dim = dim
        self.T = T
        self.alpha = create_alphas(T).to(device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)

        self.sigma = ((1 - self.alpha) * (1 - torch.roll(self.alpha_bar, 1))
                     / (1 - self.alpha_bar)) ** 0.5
        self.sigma[0] = 0. # nan due to denominator

        time_dim = 8
        self.time_net = nn.Sequential(
            nn.Linear(self.T, 16),
            nn.ELU(),
            nn.Linear(16, 8)
        )
        
        self.noise_net =  nn.Sequential(
            nn.Linear(dim + time_dim, 50),
            nn.ELU(),
            nn.Linear(50, 100),
            nn.ELU(),
            nn.Linear(100, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, dim)
        )

        self.device = device
        self.to(device)

    def forward(self, x_t, timesteps):
        """ Predict noise """
        t_emb = F.one_hot(timesteps, self.T).float()
        t_emb = self.time_net(t_emb)
        x_t_time = torch.cat([x_t, t_emb], dim=1)
        return self.noise_net(x_t_time)

    def _denoising_step(self, x_t, t):
        """
        Predict approximation of q(x_t-1 | x_t, x_o) with p(x_t-1 | x_t, t) and sample
        from it.
        """
        timesteps = torch.full((len(x_t),), t).to(self.device)
        eps_pred = self(x_t, timesteps)
        eps_pred *= (1 - self.alpha[t]) / ((1 - self.alpha_bar[t])**0.5 + 1e-31)
        mean =  1 / (self.alpha[t] ** 0.5) * (x_t - eps_pred)
        return mean + self.sigma[t] * torch.randn_like(x_t)
        
    def sample(self, n):
        with torch.no_grad():
            x_t = torch.randn((n, self.dim)).to(self.device)
            for t in range(self.T-1, -1, -1):
                x_t = self._denoising_step(x_t, t)
            return x_t * 3. # reverse normalization