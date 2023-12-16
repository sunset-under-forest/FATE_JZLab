import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torchvision.utils import save_image
from const import *
from CGAN import Generator, Discriminator
import torchvision.utils as vutils


def run(train_loader, batch_size, epochs, lr, classes, channels, img_size, noisy_dim):
    log_interval = 5

    # Setup the generator and the discriminator
    netG = Generator(classes, channels, img_size, noisy_dim).to(device)
    print(netG)
    netD = Discriminator(classes, channels, img_size, noisy_dim).to(device)
    print(netD)

    # Setup Adam optimizers for both G and D
    optim_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    netG.train()
    netD.train()
    dataloader = train_loader

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            real_label = torch.full((batch_size, 1), 1., device=device)
            fake_label = torch.full((batch_size, 1), 0., device=device)

            # Train G
            netG.zero_grad()
            z_noise = torch.randn(batch_size, noisy_dim, device=device)
            x_fake_labels = torch.randint(0, classes, (batch_size,), device=device)
            x_fake = netG(z_noise, x_fake_labels)
            y_fake_g = netD(x_fake, x_fake_labels)
            g_loss = netD.loss(y_fake_g, real_label)
            g_loss.backward()
            optim_G.step()

            # Train D
            netD.zero_grad()
            y_real = netD(data, target)
            d_real_loss = netD.loss(y_real, real_label)
            y_fake_d = netD(x_fake.detach(), x_fake_labels)
            d_fake_loss = netD.loss(y_fake_d, fake_label)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f}'.format(
                    epoch, batch_idx, len(dataloader),
                    d_loss.mean().item(),
                    g_loss.mean().item()))

    return netG


def main():
    # run()
    ...


if __name__ == '__main__':
    main()
