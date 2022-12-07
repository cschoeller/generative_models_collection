import numpy as np
from scipy.stats import multivariate_normal


class ConstGaussianMixture():
    """
    Hard-coded gaussian mixture model with three components to generate ground truth data.
    """

    def __init__(self):
        """ Define mixture components, only component 3 has covariances. """
        self.num_comps = 3
        self.mu = np.array([[4,5],[5,2],[1,3]])
        self.weights = np.array([0.35, 0.35, 0.3])
        cov1 = np.diag(np.array([0.7, 0.7]))
        cov2 = np.diag(np.array([1.25, 1.5]))
        cov2[0][1] = 1
        cov2[1][0] = 1
        cov3 = np.diag(np.array([0.2, 1.]))
        self.covs = [cov1, cov2, cov3]

    def _get_normal(self, i):
        """ Initialize a normal distribution with the parameters of component i. """
        mu, cov = self.mu[i], self.covs[i]
        return multivariate_normal(mean=mu, cov=cov)

    def __call__(self, x):
        """ Compute likelihood of x as p(x) = \sum_i w_i * p_i(x). """
        likelihood = 0
        for i in range(self.num_comps):
            w = self.weights[i]
            normal = self._get_normal(i)
            likelihood += w * normal.pdf(x)
        return likelihood

    def sample(self, n):
        """
        Sample n data points from the GMM. A single sample is drawn by first sampling a gaussian
        component according to the component weights and then drawing a sample from that gaussian.
        To accelerate this algorithm, at first for each sample a component is selected, then from
        each component the number of required samples is drawn accordingly, and finally all samples
        are uniformly shuffled to not have them grouped by component.
        """
        samples = []
        sampled_components = np.random.choice(self.num_comps, n, p=self.weights.tolist())
        for i in range(self.num_comps):
            num_component_samples = (sampled_components == i).sum()
            normal = self._get_normal(i)
            comp_samples = np.array(normal.rvs(size=num_component_samples)).reshape((-1,2))
            samples.append(comp_samples)
        samples = np.concatenate(samples)
        np.random.shuffle(samples)
        return samples