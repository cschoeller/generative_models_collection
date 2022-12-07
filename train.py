"""
Implementations of various generative models for sampling and density estimation.

Further interesting models:
- Diffusion models
- Autoregressive transformers
- RealNVP for fast sampling and inference, less expressive
- Sum-of-Squares Poldsynomial Flows (but maybe with polynomials of degree <2
  not so expressive, and otherwise no analytic inverse)
"""

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns; sns.set()

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.utils.data as Data

import losses
from adversarial_trainer import AdversarialTrainer
from models.gan import GANDiscriminator, GANGenerator
from models.made import GaussianMADE
from models.maf import MAF, IAF
from models.vae import VariationalAutoencoder
from models.spline_flow import NeuralSplineFlow
from datasets.planar_datasets import load_dataset, DatasetType
from pycandle.training.model_trainer import ModelTrainer


class Config:
    lr = 0.001
    epochs = 30
    gan_epochs = 200
    batch_size = 64
    model_samples = 50000
    device = 0 # gpu
    dataset_type = DatasetType.FACE
    dataset_size = 50000


def load_model(model, checkpoint_path):
        print(f"Loading model '{checkpoint_path}' from drive...")
        loaded_state_dict = torch.load(checkpoint_path)['model_state_dict']
        model.load_state_dict(loaded_state_dict)

def store_model(model, checkpoint_path):
    model.eval()
    model_state_dict = model.state_dict()
    torch.save({'arch' : model.__class__.__name__,
                'model_state_dict' : model_state_dict,
                }, checkpoint_path)

def train_model(dataset, model, loss, checkpoint_path):
    # load model from drive if it was trained earlier
    if os.path.isfile(checkpoint_path):
       load_model(model, checkpoint_path)
       return
    #train model
    print(f"Training model {model.__class__.__name__}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10) # reset lr every n epochs
    train_loader = Data.DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    model_trainer = ModelTrainer(model, optimizer, loss, Config.epochs, train_loader, custom_model_eval=True, device=Config.device, scheduler=scheduler)
    model_trainer.start_training()
    store_model(model, checkpoint_path)

def train_gan(dataset, gen, disc, checkpoint_path):
    # load model from disc
    if os.path.isfile(checkpoint_path):
        print(f"Loading model '{checkpoint_path}' from drive...")
        generator_state_dict = torch.load(checkpoint_path)['generator_state_dict']
        discriminator_state_dict = torch.load(checkpoint_path)['discriminator_state_dict']
        gen.load_state_dict(generator_state_dict)
        disc.load_state_dict(discriminator_state_dict)
        return

    print(f"Training model GAN...")
    train_loader = Data.DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True, num_workers=1)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=Config.lr)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=Config.lr)
    models, optimizers = [gen, disc], [opt_gen, opt_disc]
    trainer = AdversarialTrainer(models, optimizers, nn.BCELoss(), Config.gan_epochs, train_loader, device=Config.device)
    trainer.start_training()

    # save model
    gen.eval()
    disc.eval()
    gen_state_dict, disc_state_dict = gen.state_dict(), disc.state_dict()
    torch.save({'generator_state_dict' : gen_state_dict,
                'discriminator_state_dict' : disc_state_dict,
                }, checkpoint_path)

def plot_training_datasets():
    out_folder = Path("dataset_plots")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for ds_type in DatasetType:
        output_path = out_folder / f"{ds_type.value}.png"
        if os.path.exists(output_path):
            continue
        print(f"Plotting dataset {ds_type}")
        dataset = load_dataset(ds_type, 20000)
        data = dataset.data
        plt.axis('equal')
        plt.scatter(data[:,0],data[:,1], color='red', alpha=0.6, s=0.3)
        plt.savefig(output_path)
        plt.close()

def plot_all_samples(dataset, model_samples):
    print("Plotting distributions...")
    gt_data = dataset.data.numpy()
    X, Y = gt_data[:,0], gt_data[:,1]
    x_lims = np.array([np.min(X), np.max(X)]) * 1.3
    y_lims = np.array([np.min(Y), np.max(Y)]) * 1.3

    f, axes = plt.subplots(1, len(model_samples) + 1, figsize=(24, 7), sharex=True)
    [axis.set(xlim=x_lims.tolist(), ylim=y_lims.tolist()) for axis in axes]

    sns.despine(left=True)
    #sns.kdeplot(x=gt_data[:,0], y=gt_data[:,1], cmap="Reds", fill=True, thresh=0.05,
    #            ax=axes[0]).set_title("Ground Truth")
    sns.scatterplot(x=gt_data[:,0], y=gt_data[:,1], color="red", s=0.5, ax=axes[0]).set_title("Ground Truth")
    for i, (model_name, samples) in enumerate(model_samples.items(), start=1):
        samples = samples.cpu().detach().numpy()
        #ns.kdeplot(x=samples[:,0], y=samples[:,1], cmap="Reds", fill=True,
        #            thresh=0.05, ax=axes[i]).set_title(model_name)
        sns.scatterplot(x=samples[:,0], y=samples[:,1], color="red", s=0.5, ax=axes[i]).set_title(model_name)

    plt.tight_layout()
    plt.savefig(f"distributions_{Config.dataset_type.value}.png")
    #plt.show()
    #plt.close()


def main():
    plot_training_datasets()

    # load specified dataset
    target_dataset = load_dataset(Config.dataset_type, Config.dataset_size)
    checkpoint_path = Path(f"checkpoints/{str(Config.dataset_type.value)}/")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model_samples = {}

    vae = VariationalAutoencoder(nin=2, hiddens=[8, 16, 32, 64], device=Config.device)
    train_model(target_dataset, vae, losses.vae_loss, checkpoint_path / "vae_checkpoint.pt")
    model_samples["VAE"] = vae.sample(Config.model_samples)

    gan_gen = GANGenerator(noise_size=8, nh=[24, 24, 24], nout=2, device=0)
    gan_disc = GANDiscriminator(nin=2, nh=[24, 24, 24], device=0)
    train_gan(target_dataset, gan_gen, gan_disc, checkpoint_path / "gan_checkpoint.pt")
    model_samples["GAN"] = gan_gen.sample(Config.model_samples)

    made = GaussianMADE(nin=2, hidden_sizes=[16, 32, 16])
    train_model(target_dataset, made, losses.made_loss, checkpoint_path / "made_checkpoint.pt")
    model_samples["MADE"] = made.sample(Config.model_samples)

    maf = MAF(nin=2, num_mades=5, nh=32, device=Config.device, batchnorm=False)
    train_model(target_dataset, maf, losses.maf_loss, checkpoint_path / "maf_checkpoint.pt")
    model_samples["MAF"] = maf.sample(Config.model_samples)

    iaf = IAF(nin=2, num_mades=5, nh=32, device=Config.device)
    train_model(target_dataset, iaf, losses.iaf_loss, checkpoint_path / "iaf_checkpoint.pt")
    model_samples["IAF"] = iaf.sample(Config.model_samples)

    nsf = NeuralSplineFlow(nin=2, n_layers=8, K=10, B=6, hidden_dim=8, device=Config.device)
    train_model(target_dataset, nsf, losses.nll, checkpoint_path / "nsf_checkpoint.pt")
    model_samples["NSF"] = nsf.sample(Config.model_samples)

    plot_all_samples(target_dataset, model_samples)


if __name__ == "__main__":
    main()
