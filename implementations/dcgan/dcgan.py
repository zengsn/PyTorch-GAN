import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

from wavelet_denoise2 import get_wavelet
from wavelet_denoise2 import WaveletLayer2

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# Support different device
device = torch.device('cpu')  # default
cuda = True if torch.cuda.is_available() else False
mac_m1 = True if torch.backends.mps.is_available() else False
if cuda:
    device = torch.device('cuda:0')
elif mac_m1: # Macbook M1 GPU
    device = torch.device('mps')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, wavelet=None, wave_dropout=0.0):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.wavelet = wavelet
        if wavelet is not None:
            self.wave_dropout = wave_dropout
            self.wavelet_layer = WaveletLayer2()

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        if self.wavelet is not None:
            img0 = img
            img1 = self.wavelet_layer(img0)
            # img = img1.subtract(img0)
            # img = img1.add(img0)
            # img[img > 255] = 255
            img = img0 - 0.01*img1
        return img

    def wavelet_loss(self):
        return self.wavelet_layer.wavelet_loss() if self.wavelet is not None else None


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
# random init
# init_wavelet = ProductFilter(
#     torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
#     torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
#     torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
#     torch.rand(size=[6], requires_grad=True) / 2 - 0.25,
# )

generator = Generator(
    wavelet=get_wavelet(78)
)
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    print("Using CUDA GPU!")
elif mac_m1:
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    print("Using mps by Macbook M1 GPU!")

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        # valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        # if cpu or cuda or mac_m1:
        valid =Variable(torch.ones((imgs.shape[0],1), device=device, requires_grad=False))
        fake =Variable(torch.zeros((imgs.shape[0],1), device=device, requires_grad=False))

        # Configure input
        # real_imgs = Variable(imgs.type(Tensor))
        # if cpu or cuda or mac_m1:
        real_imgs = Variable(torch.tensor(imgs, device=device))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # if cpu or cuda or mac_m1:
        arr_int32 = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)).astype(np.float32)
        z = Variable(torch.tensor(arr_int32, device=device))  # float32

        # Generate a batch of images
        gen_imgs = generator(z)
        # w_loss = generator.wavelet_loss()

        # Loss measures generator's ability to fool the discriminator
        g_disc = discriminator(gen_imgs)
        g_loss = adversarial_loss(g_disc, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_disc = discriminator(real_imgs)
        real_loss = adversarial_loss(real_disc, valid)
        fake_disc = discriminator(gen_imgs.detach())
        fake_loss = adversarial_loss(fake_disc, fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            image_name = "images/%d.png" if generator.wavelet is None else "images/%d-w.png"
            save_image(gen_imgs.data[:25], image_name % batches_done, nrow=5, normalize=True)
