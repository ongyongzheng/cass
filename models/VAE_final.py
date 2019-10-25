import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from models.baseline import Generator

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

  def load_model(self, path):
    for i in range(self.num_networks):
      # load
      vae_path = path + 'vae_' + str(i)
      self.vae[i].load_state_dict(torch.load(vae_path))

