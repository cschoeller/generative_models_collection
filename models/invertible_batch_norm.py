import torch
import torch.nn as nn


class InvertibleBatchNorm(nn.Module):
    """
    Implements batch normalization as an invertible flow function. The updates with
    momentum are just one variant. It is also possible to count the batches and divide mu
    and var before every transformation. Ideally, the running mu and var should be used
    for standardization, not the batch one. But this lead to technical issues.
    """

    def __init__(self, num_features, momentum=0.2, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(num_features, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(num_features, requires_grad=True))
        self.register_buffer('mu', torch.zeros(num_features)) # running stats
        self.register_buffer('var', torch.ones(num_features))

    def forward(self, x, c=None):
        mu, var = self.update_running_stats(x)
        std = var.sqrt()
        x = (x - mu) / std # element-wise standardization
        x = torch.exp(self.alpha) * x + self.beta # scale and shift
        logdet = (self.alpha - 0.5 * torch.log(var)).sum(-1)
        logdet = logdet.repeat(x.size(0)) # for each batch element
        return x, logdet

    def backwards(self, x):
        mu, var = self.update_running_stats(x)
        std = var.sqrt()
        x = (x - self.beta) / torch.exp(self.alpha) # inverse scale and shift
        x = (x * std) + mu # inverse standardization
        logdet = (- self.alpha + 0.5 * torch.log(var)).sum(-1)
        logdet = logdet.repeat(x.size(0)) # for each batch element
        return x, logdet

    def inverse(self, x, c=None):
        return self.backwards(x)

    def update_running_stats(self, x):
        if not self.training:
            return self.mu, self.var
        batch_mu = x.mean(0)
        batch_var = ((x - batch_mu)**2).mean(0) + self.eps
        self.mu = (1. - self.momentum) * self.mu + self.momentum * batch_mu # update
        self.var = (1. - self.momentum) * self.var + self.momentum * batch_var
        return batch_mu, batch_var
        
