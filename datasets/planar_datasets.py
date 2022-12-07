""" Various 2-dim datasets useful for density estimation, partially adapted from https://github.com/bayesiains/nsf. """

from abc import abstractmethod
import math
import random
from enum import Enum
from pathlib import Path
import io

import numpy as np
import torch
from sklearn import datasets
from skimage import io, color, transform
import matplotlib.pyplot as plt
from torch import distributions
from torch.utils.data import Dataset

from .gaussian_mixture import ConstGaussianMixture


class _PlanarDataset(Dataset):
    """ Abstract dataset type for planar datasetse. """

    def __init__(self, num_points, flip_axes=False):
        self.num_points = num_points
        self.flip_axes = flip_axes
        self.data = None
        self.reset()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def reset(self):
        self.data = self._create_data()
        self._normalize_data()
        if self.flip_axes:
            x1 = self.data[:, 0]
            x2 = self.data[:, 1]
            self.data = torch.stack([x2, x1]).t()
    
    def _normalize_data(self):
        mean = torch.mean(self.data, dim=0)
        centered_data = self.data - mean
        x_max = centered_data[:,0].max()
        self.data = (centered_data / x_max) * 3.

    @abstractmethod
    def _create_data(self):
        ...

class _GaussianDataset(_PlanarDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2 = 0.5 * torch.randn(self.num_points)
        return  torch.stack((x1, x2)).t()

class _CrescentDataset(_PlanarDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = 0.5 * x1 ** 2 - 1
        x2_var = torch.exp(torch.Tensor([-2]))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        return  torch.stack((x2, x1)).t()

class _CrescentCubedDataset(_PlanarDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = 0.2 * x1 ** 3
        x2_var = torch.ones(x1.shape)
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        return torch.stack((x2, x1)).t()

class _SineWaveDataset(_PlanarDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sin(5 * x1)
        x2_var = torch.exp(-2 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        return torch.stack((x1, x2)).t()

class _AbsDataset(_PlanarDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.abs(x1) - 1.
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        return torch.stack((x1, x2)).t()

class _SignDataset(_PlanarDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sign(x1) + x1
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        return  torch.stack((x1, x2)).t()

class _FourCircles(_PlanarDataset):
    def __init__(self, num_points, flip_axes=False):
        super().__init__(num_points, flip_axes)

    def create_circle(self, num_per_circle, std=0.1):
        u = torch.rand(num_per_circle)
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        num_per_circle = self.num_points // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        data = torch.cat(
            [self.create_circle(num_per_circle) - torch.Tensor(center)
             for center in centers]
        )
        return data

class _DiamondDataset(_PlanarDataset):
    def __init__(self, num_points, flip_axes=False):
        self.width = 20
        self.bound = 2.5
        self.std = 0.04
        super().__init__(num_points, flip_axes)

    def _create_data(self, rotate=True):
        means = np.array([
            (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
            for x in np.linspace(-self.bound, self.bound, self.width)
            for y in np.linspace(-self.bound, self.bound, self.width)
        ])
        covariance_factor = self.std * np.eye(2)

        index = np.random.choice(range(self.width ** 2), size=self.num_points, replace=True)
        noise = np.random.randn(self.num_points, 2)
        self.data = means[index] + noise @ covariance_factor
        if rotate:
            rotation_matrix = np.array([
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2)]
            ])
            self.data = self.data @ rotation_matrix
        data = self.data.astype(np.float32)
        return torch.Tensor(data)

class _TwoSpiralsDataset(_PlanarDataset):
    def _create_data(self):
        n = torch.sqrt(torch.rand(self.num_points // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(self.num_points // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(self.num_points // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
        return x / 3 + torch.randn_like(x) * 0.1

class _GridDataset(_PlanarDataset):
    def _create_data(self):
        num_points_per_axis = int(math.sqrt(self.num_points))
        bounds = [[-3.,3.],[-3.,3.]]
        x = np.linspace(bounds[0][0], bounds[0][1], num_points_per_axis)
        y = np.linspace(bounds[1][0], bounds[1][1], num_points_per_axis)
        X_, Y_ = np.meshgrid(x, y)
        data = np.vstack([X_.flatten(), Y_.flatten()]).T
        return torch.tensor(data).float()

class _CheckerboardDataset(_PlanarDataset):
    def _create_data(self):
        x1 = torch.rand(self.num_points) * 4 - 2
        x2_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        return torch.stack([x1, x2]).t() * 2

class _MoonsDataset(_PlanarDataset):
    def _create_data(self):
        moons = datasets.make_moons(n_samples=self.num_points, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)

class _GaussianMixtureDataset(_PlanarDataset):
    def _create_data(self):
        return torch.from_numpy(ConstGaussianMixture().sample(self.num_points)).float() # sample points

class _FaceDataset(_PlanarDataset):
    def __init__(self, num_points, name='jobs', flip_axes=False):
        self.name = name
        super().__init__(num_points, flip_axes)

    def _create_data(self):
        path = Path(f"./datasets/{self.name}.jpg")
        try:
            image = io.imread(path)
        except FileNotFoundError:
            raise RuntimeError(f"Image not found: {path}")
        image = color.rgb2gray(image)
        image = transform.resize(image, (image.shape[0] // 4, image.shape[1] // 4),
                       anti_aliasing=True)

        # create dataset
        pos_grid = np.array([
            (x, y) for x in range(image.shape[0]) for y in range(image.shape[1])
        ])
        rotation_matrix = np.array([
            [0, -1],
            [1, 0]
        ])
        pos_grid = np.matmul(pos_grid, rotation_matrix) # rotate axes
        pixel_weights = image.flatten()
        pixel_weights = np.ones_like(pixel_weights) - pixel_weights
        data = (pos_grid[pixel_weights > 0.5]).astype(np.float64) # watershed
        data = data[random.choices(np.arange(0, data.shape[0]), k=self.num_points)] # uniform sub-sample
        data = data + (np.random.rand(self.num_points, 2) * 2.) # dequantize with std 2
        return torch.tensor(data).float()


class DatasetType(str, Enum):
    FACE = "face"
    GAUSS = "gauss"
    CRES = "cres"
    CRESCUBE = "crescube"
    SINE = "sine"
    ABS = "abs"
    SIGN = "sign"
    CIRC = "circ"
    DIAMOND = "diamond"
    SPIRAL = "spiral"
    GRID = "grid"
    CHECKER = "checker"
    MOON = "moon"
    GMM = "gmm"

_DATASET_MAPPING = {
    DatasetType.FACE : _FaceDataset,
    DatasetType.GAUSS : _GaussianDataset,
    DatasetType.CRES : _CrescentDataset,
    DatasetType.CRESCUBE : _CrescentCubedDataset,
    DatasetType.SINE : _SineWaveDataset,
    DatasetType.ABS : _AbsDataset,
    DatasetType.SIGN : _SignDataset,
    DatasetType.CIRC : _FourCircles,
    DatasetType.DIAMOND : _DiamondDataset,
    DatasetType.SPIRAL : _TwoSpiralsDataset,
    DatasetType.GRID : _GridDataset,
    DatasetType.CHECKER : _CheckerboardDataset,
    DatasetType.MOON : _MoonsDataset,
    DatasetType.GMM : _GaussianMixtureDataset
}

def load_dataset(dataset_type, dataset_size):
    """ Return requested dataset with specified size. """
    return _DATASET_MAPPING[dataset_type](dataset_size)