"""
Implements the following generative model:

"Auto-Encoding Variational Bayes", Kingma et al., Dec 2013 (VAE)
https://arxiv.org/abs/1312.6114v1
"""


import torch
import torch.nn as nn
#import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Learns a normal distributed latent variable embedding from which it is easy to
    sample and decode. This implementation assumes a symmetric (except for the output
    layer) encoder and decoder.
    """
    
    def __init__(self, nin, hiddens, device=0):
        super().__init__()

        # construct encoder network
        hs_enc = [nin] + hiddens[:-1]
        self.latent_size = hiddens[-1]
        self.enc = []
        for (h1, h2) in zip(hs_enc, hs_enc[1:]):
            self.enc.extend([nn.Linear(h1, h2), nn.ReLU()])
        self.enc = nn.Sequential(*self.enc)
        self.enc_mu = nn.Linear(hs_enc[-1], self.latent_size)
        self.enc_cov = nn.Linear(hs_enc[-1], self.latent_size)

        # construct decoder network
        hs_dec = list(reversed([nin] + hiddens))
        self.dec = []
        for (h1, h2) in zip(hs_dec, hs_dec[1:]):
            self.dec.extend([nn.Linear(h1, h2), nn.ReLU()])
        self.dec.pop() # remove last relu
        self.dec = nn.Sequential(*self.dec)

        self.device = device
        self.to(self.device)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick allows differentiable sampling. The sample from the
        standard normal is considered a constant and then transformed into the
        target gaussian.
        """
        std = torch.exp(0.5*logvar) # equals sqrt(var)
        eps = torch.randn_like(std)
        return std * eps + mu

    def encode(self, x):
        """ Encode the input x into the latent posterior p(z|x). """
        x = self.enc(x)
        mu = self.enc_mu(x)
        logvar = self.enc_cov(x)
        return mu, logvar

    def decode(self, z):
        """ Decode the latent variable f(z) -> x. """
        return self.dec(z)

    def forward(self, x):
        """
        First the encoder predicts parameters of the normal latent distribution
        and then a sample from it is drawn and decoded. This is used for training.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n):
        """ Draw samples from the standard normal prior and decode it. """
        samples = torch.zeros([n, self.latent_size]).to(self.device)
        z = torch.randn_like(samples)
        return self.decode(z)