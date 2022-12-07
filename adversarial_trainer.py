import torch
import torch.nn as nn
from pycandle.training.utils import recursive_to_cuda


class AdversarialTrainer():
    """
    Model trainer for GANs, i.e., alternately backpropagating and optimizing the generator
    and the discriminator.
    """

    def __init__(self, models, optimizers, loss, epochs, train_data_loader, device=0):
        assert(len(models) == 2)
        assert(len(optimizers) == 2)
        self.generator, self.discriminator = models
        self.gen_optimizer, self.disc_optimizer = optimizers
        self.loss = loss # discriminator loss
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.device = device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def start_training(self):
        for epoch in range(1, self.epochs + 1):
            self._epoch_step(epoch)

    def _epoch_step(self, epoch):
        running_gen_loss = 0
        running_disc_loss_real = 0
        running_disc_loss_fake = 0

        for step, batch in enumerate(self.train_data_loader):
            batch = recursive_to_cuda(batch, self.device) # move to GPU

            # create labels
            true_labels = torch.ones(batch.size(0), 1).float().to(self.device)
            false_labels = torch.zeros(batch.size(0), 1).float().to(self.device)

            # generate samples
            self.gen_optimizer.zero_grad()
            noise = torch.randn(batch.size(0), self.generator.noise_dims).to(self.device)
            generated_data = self.generator(noise)

            # generator training step
            generator_loss = torch.tensor([0])
            if epoch > 1: # discriminator head-start to stabilize training
                # make the discriminator classify generated data
                generator_discriminator_labels = self.discriminator(generated_data)

                # train generator such that it 'fools' the discriminator
                generator_loss = self.loss(generator_discriminator_labels, true_labels)
                generator_loss.backward()
                self.gen_optimizer.step()

            # discriminator training step; learns to differentiate true and fake data
            self.disc_optimizer.zero_grad()
            data_discriminator_out = self.discriminator(batch) # loss on real data
            data_discriminator_loss = self.loss(data_discriminator_out, true_labels)
            generator_discriminator_out = self.discriminator(generated_data.detach()) # loss on generated
            generator_discriminator_loss = self.loss(generator_discriminator_out, false_labels)
            discriminator_loss = (data_discriminator_loss + generator_discriminator_loss) / 2.
            discriminator_loss.backward()
            self.disc_optimizer.step()

            running_gen_loss += generator_loss.item()
            running_disc_loss_real += data_discriminator_loss.item()
            running_disc_loss_fake += generator_discriminator_loss.item()

            print(f"epoch {epoch}   batch {step}/{len(self.train_data_loader) - 1}   generator_loss:" \
                 f"{running_gen_loss/(step+1):.4f}   discriminator_loss_real: {running_disc_loss_fake/(step+1):.4f}"\
                 f"   discr_loss_fake: {running_disc_loss_fake/(step+1):.4f}")
