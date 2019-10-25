import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, nz=100):
        super(Encoder, self).__init__()
        kernel_size = 5
        padding = 2

        self.nz = nz
        # conv1:
        self.conv1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        )
        # conv2:
        self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )
        # conv3:
        self.conv3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        )
        # conv4:
        self.conv4 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
        )
        # conv5:
        self.conv5 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
        )
        # fc6:
        self.fc6 = nn.Sequential(
        nn.Linear(512 * 16 * 4 , 1024, bias=False),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True)
        )

        # mu: 1024 -> nz
        self.mu = nn.Linear(1024, nz, bias=False)

        # logvar: 1024 -> nz
        self.logvar = nn.Linear(1024, nz, bias=False)

    def forward(self, x, batch_size=64):
        x = x.view(-1, 1, 512, 128)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(batch_size, -1)
        h = self.fc6(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

class Decoder(nn.Module):
  def __init__(self, nz):
    super(Decoder, self).__init__()
    kernel_size = 5
    padding = 2

    # input params
    self.nz = nz
    # fc1:
    self.fc1 = nn.Sequential(
      nn.Linear(nz, 512 * 16 * 4, bias=False),
      nn.BatchNorm1d(512 * 16 * 4),
      nn.ReLU(inplace=True)
    )
    # deconv2:
    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    # deconv3:
    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    # deconv4:
    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    # deconv5:
    self.deconv5 = nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
    )
    # conv6:
    self.conv6 = nn.Sequential(
      nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(1),
      nn.Sigmoid()
    )

  def forward(self, z):
    xhat = self.fc1(z)
    xhat = xhat.view(-1, 512, 16, 4)
    xhat = self.deconv2(xhat)
    xhat = self.deconv3(xhat)
    xhat = self.deconv4(xhat)
    xhat = self.deconv5(xhat)
    mask = self.conv6(xhat)
    return mask

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    kernel_size = 5
    padding = 2
    # conv1:
    self.conv1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.2)
    )
    # conv2:
    self.conv2 = nn.Sequential(
    nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2)
    )
    # conv3:
    self.conv3 = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2)
    )
    # conv4:
    self.conv4 = nn.Sequential(
    nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2)
    )
    # conv5:
    self.conv5 = nn.Sequential(
    nn.Conv2d(256, 512, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2)
    )
    # fc6:
    self.fc6 = nn.Sequential(
    nn.Linear(512 * 16 * 4, 1024, bias=False),
    nn.BatchNorm1d(1024),
    nn.LeakyReLU(0.2)
    )
    # fc7: 1024 -> 1
    self.fc7 = nn.Sequential(
    nn.Linear(1024, 1),
    )


  def forward(self, x):
    x = torch.log1p(x)
    # remove last frequency bin
    x = x[:,:-1,:]
    x = x.view(-1, 1, 512, 128)
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    f = self.conv4(f)
    f = self.conv5(f)
    f = f.view(-1, 512 * 16 * 4)
    f = self.fc6(f)
    o = self.fc7(f)
    return o

  def feature(self, x):
    x = torch.log1p(x)
    # remove last frequency bin
    x = x[:,:-1,:]
    x = x.view(-1, 1, 512, 128)
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    return f.view(-1, 128 * 64 * 16)

  def save_model(self, path):
    torch.save(self.state_dict(), path)

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()

        # input params
        self.nz = nz

        self.encoder = Encoder(nz)
        self.decoder = Decoder(nz)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = torch.randn_like(std)
          return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, x):
        x_in = torch.log1p(x)
        # remove last frequency bin
        x_in = x_in[:,:-1,:]
        mu, logvar = self.encoder(x_in, x.shape[0])
        z = self.reparameterize(mu, logvar)
        mask = self.decoder(z)
        # pad mask
        mask = F.pad(mask, (0, 0, 1, 0), value=0.5)
        mask = torch.squeeze(mask)
        return x * mask, mu, logvar

    def encode(self, x):
        x_in = torch.log1p(x)
        # remove last frequency bin
        x_in = x_in[:,:,:-1,:]
        mu, logvar = self.encoder(x_in, x.shape[0])
        return mu

    def generate(self, z):
        self.eval()
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x):
        self.eval()
        x_in = torch.log1p(x)
        # remove last frequency bin
        x_in = x_in[:,:-1,:]
        mu, logvar = self.encoder(x_in, x.shape[0])
        mask = self.decoder(mu)
        # pad mask
        mask = F.pad(mask, (0, 0, 1, 0), value=0.5)
        mask = torch.squeeze(mask)
        return x * mask

    def save_model(self, path):
        torch.save(self.state_dict(), path)

