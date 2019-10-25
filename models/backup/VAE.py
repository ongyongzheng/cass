import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, nz=100):
        super(Encoder, self).__init__()

        self.nz = nz
        # conv1: 3 x 64 x 64 -> 32 x 64 x 64
        self.conv1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        )
        # conv2: 32 x 64 x 64 -> 64 x 32 x 32
        self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )
        # conv3: 64 x 32 x 32 -> 128 x 16 x 16
        self.conv3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        )
        # conv4: 128 x 16 x 16 -> 256 x 8 x 8
        self.conv4 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
        )
        # conv5: 256 x 8 x 8 -> 512 x 4 x 4
        self.conv5 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
        )
        # fc6: 512 * 4 * 4 -> 1024
        self.fc6 = nn.Sequential(
        nn.Linear(512 * 32 * 8 , 1024, bias=False),
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

    # input params
    self.nz = nz
    # fc1: nz -> 512 * 4 * 4
    self.fc1 = nn.Sequential(
      nn.Linear(nz, 512 * 32 * 8, bias=False),
      nn.BatchNorm1d(512 * 32 * 8),
      nn.ReLU(inplace=True)
    )
    # deconv2: 512 x 4 x 4 -> 256 x 8 x 8
    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    # deconv3: 256 x 8 x 8 -> 128 x 16 x 16
    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    # deconv4: 128 x 16 x 16 -> 64 x 32 x 32
    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    # deconv5: 64 x 32 x 32 -> 32 x 64 x 64
    self.deconv5 = nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
    )
    # conv6: 32 x 64 x 64 -> 3 x 64 x 64
    self.conv6 = nn.Sequential(
      nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(1),
      nn.Sigmoid()
    )
    """# fc7: 3 x 64 x 64 -> 8514
    self.fc7 = nn.Sequential(
    nn.Linear(1 * 528 * 32, 11286, bias=False),
    nn.Sigmoid()
    )"""

  def forward(self, z):
    xhat = self.fc1(z)
    xhat = xhat.view(-1, 512, 32, 8)
    xhat = self.deconv2(xhat)
    xhat = self.deconv3(xhat)
    xhat = self.deconv4(xhat)
    xhat = self.deconv5(xhat)
    mask = self.conv6(xhat)
    #xhat = xhat.view(-1, 1 * 528 * 32)
    #mask = self.fc7(xhat)
    return mask

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

class VAE(object):
  def __init__(self,
      num_networks,
      learning_rate,
      device,
      Nz
  ):
    self.num_networks = num_networks
    self.learning_rate = learning_rate
    self.device = device
    self.Nz = Nz

  def build_model(self):
    self.vae = []
    self.enc_optim = []
    self.dec_optim = []

    for i in range(self.num_networks):
      ae = Generator(self.Nz).to(self.device)
      self.vae.append(ae)

      # create optimizers
      enc_o = optim.Adam(
        self.vae[i].encoder.parameters(),
        lr= self.learning_rate,
        betas=(0.5, 0.999))
      dec_o = optim.Adam(
        self.vae[i].decoder.parameters(),
        lr= self.learning_rate,
        betas=(0.5, 0.999))
      self.enc_optim.append(enc_o)
      self.dec_optim.append(dec_o)

    # create history data
    self.history = {}
    self.history['enc_loss'] = []
    self.history['dec_loss'] = []

  def train(self, x, components):
    # set to train mode
    for i in range(self.num_networks):
      self.vae[i].train()

    # set up loss functions
    def l1_loss(input, target):
      return torch.mean(torch.abs(input - target))
    def l2_loss(input, target):
      return torch.mean((input - target).pow(2))

    enc_losses = 0
    dec_losses = 0

    batch_size = x.shape[0]
    # reset gradients
    for i in range(self.num_networks):
      self.enc_optim[i].zero_grad()
      self.dec_optim[i].zero_grad()

    x_real = Variable(torch.from_numpy(x)).to(self.device)
    cp_real = []
    for i in components:
      cp_real.append(Variable(torch.from_numpy(i)).to(self.device))

    for i in range(self.num_networks):
      # train encoder
      cp_fake, mu, logvar = self.vae[i](x_real)
      # l_prior
      l_prior = 1e-2 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
      # l_recon
      l_recon = 5 * l1_loss(cp_fake, cp_real[i])
      # enc_loss
      enc_loss = l_prior + l_recon
      enc_loss.backward()
      self.enc_optim[i].step()
      enc_losses += enc_loss

      # train decoder
      cp_fake, mu, logvar = self.vae[i](x_real)
      # l_like
      l_recon = 5 * l1_loss(cp_fake, cp_real[i])
      # dec_loss
      dec_loss = l_recon
      dec_loss.backward()
      self.dec_optim[i].step()
      dec_losses += dec_loss

    enc_losses /= self.num_networks
    dec_losses /= self.num_networks
    self.history['enc_loss'].append(enc_losses)
    self.history['dec_loss'].append(dec_losses)
    return enc_losses, dec_losses

  def test(self, x):
    # set to eval mode
    for i in range(self.num_networks):
      self.vae[i].eval()

    result = []
    x_real = Variable(torch.from_numpy(x)).to(self.device)
    for i in range(self.num_networks):
      result.append(self.vae[i](x_real)[0].cpu().detach().numpy())
    return result

  def save_model(self, path):
    for i in range(self.num_networks):
      # save vae
      vae_path = path + 'vae_' + str(i)
      self.vae[i].save_model(vae_path)

