import torch
import torch.nn as nn
import numpy as np

from torchvision.datasets import FakeData
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)


def forward_diffusion(x_0, alpha_bar, timesteps):
    """ Models function q(x_t | x_0) """
    eps_0 = torch.randn_like(x_0).to(x_0.device) # sample noise
    alpha_t_bar = alpha_bar[timesteps, None]
    x_t = torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1 - alpha_t_bar) * eps_0
    return eps_0, x_t

def create_alphas(T):
    """ Create alpha schedule """
    betas = torch.linspace(0., np.pi/5., T)
    alpha = torch.abs(torch.cos(betas)) # cosine schedule
    #beta = torch.linspace(10e-4, 0.02, T) # Ho et al. 2020
    #beta = torch.linspace(0., 0.2, T) # Ho et al. 2020
    #alpha = 1. - beta
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

    def __init__(self, dim, T, data_minmax=(-1., 1.), device=0):
        super().__init__()
        self.dim = dim
        self.T = T
        # self.vals_min, self.vals_max = data_minmax
        # self.val_mag = self.vals_max - self.vals_min
        self.alpha = create_alphas(T).to(device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)

        # TODO: verify correctness
        self.sigma = (
            (1 - self.alpha)
            * (1 - torch.roll(self.alpha_bar, 1)) / (1 - self.alpha_bar)
        ) ** 0.5
        self.sigma[0] = 0. # nan due to denominator‚

        time_dim = 8
        self.time_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 8)
        )
        
        self.noise_net =  nn.Sequential(
            nn.Linear(dim + time_dim, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, dim)
        )

        self.device = device
        self.to(device)

    # def normalize(self, x):
    #     """ Squeeze data in range [-1, 1] """
    #     return (x + self.vals_min) * (2/self.val_mag) - 1.

    # def denormalize(self, x):‚
    #     """ Revert data normalization """
    #     return (x + 1.) / (2/self.val_mag) - self.vals_min

    def forward(self, x_t, timesteps):
        """ Predict noise """
        if len(timesteps.shape) == 1:
            timesteps = timesteps.unsqueeze(1)

        t_emb = self.time_net(timesteps.float())
        x_t_time = torch.cat([x_t, t_emb], dim=1) # cat time embedding
        return self.noise_net(x_t_time)

    def _denoising_step(self, x_t, t):
        """
        Predict mean of q(x_t-1 | x_t, x_o) with p(x_t-1 | x_t, t). We don't need
        the variance as we don't want to sample from this distribution.
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
            return x_t
            

# Last major change:
# Sampling from denoising distribution. Seems it is VERY important, otherwise it seems
# net network just undoes all points noise and deterministically pushes them forward
# from the (0,0) center on the same deterministic route! This was explained badly in the
# survey paper so far. Then I also had a left over bug where I basically kept the t fixed
# in the denoising.
# Probably also model too small.


# # TODO use torch fill, cumprod, maybe roll

# # torch.set_printoptions(sci_mode=False)

# # T = 10
# # alphas = create_alphas(T)
# # print(alphas)

# # img = Image.open("../datasets/jobs.jpg")
# # img.thumbnail((256, 256))

# # convert_tensor = transforms.ToTensor()
# # img = convert_tensor(img)

# # out_folder = "noise_test/"
# # for t in range(T):
# #     e = torch.randn_like(img).to(img.device)
# #     img_t = forward_diffusion(img.unsqueeze(0), e.unsqueeze(0), alphas, torch.tensor([t]))
# #     img_t = img_t.squeeze()
# #     plt.imshow(img_t.permute(1,2,0))
# #     plt.savefig(out_folder + f"{t}_.jpg")