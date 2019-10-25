import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from models.baseline import Generator, Discriminator

class CASS(object):
  def __init__(self,
      num_networks,
      learning_rate,
      device,
      Nz,
      weight_decay=0
  ):
    self.num_networks = num_networks
    self.learning_rate = learning_rate
    self.device = device
    self.Nz = Nz
    self.weight_decay = weight_decay

  def build_model(self):
    self.generator = []
    self.discriminator = []
    self.enc_optim = []
    self.dec_optim = []
    self.dis_optim = []

    for i in range(self.num_networks):
      gen = Generator(self.Nz).to(self.device)
      dis = Discriminator().to(self.device)
      self.generator.append(gen)
      self.discriminator.append(dis)

      # create optimizers
      enc_o = optim.Adam(
        self.generator[i].encoder.parameters(),
        lr= self.learning_rate,
        weight_decay=self.weight_decay,
        betas=(0.5, 0.999))
      dec_o = optim.Adam(
        self.generator[i].decoder.parameters(),
        lr= self.learning_rate,
        weight_decay=self.weight_decay,
        betas=(0.5, 0.999))
      dis_o = optim.Adam(
        self.discriminator[i].parameters(),
        lr= self.learning_rate,
        weight_decay=self.weight_decay,
        betas=(0.5, 0.999))
      self.enc_optim.append(enc_o)
      self.dec_optim.append(dec_o)
      self.dis_optim.append(dis_o)

    # create history data
    self.history = {}
    self.history['dis_loss'] = []
    self.history['enc_loss'] = []
    self.history['dec_loss'] = []

  def train(self, x, components):
    # set to train mode
    for i in range(self.num_networks):
      self.generator[i].train()
      self.discriminator[i].train()

    # set up loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    def l1_loss(input, target):
      return torch.mean(torch.abs(input - target))
    def l2_loss(input, target):
      return torch.mean((input - target).pow(2))

    enc_losses = 0
    dec_losses = 0
    dis_losses = 0

    batch_size = x.shape[0]
    t_real = Variable(torch.ones(batch_size, 1)).to(self.device)
    t_fake = Variable(torch.zeros(batch_size, 1)).to(self.device)

    # reset gradients
    for i in range(self.num_networks):
      self.enc_optim[i].zero_grad()
      self.dec_optim[i].zero_grad()
      self.dis_optim[i].zero_grad()

    x_real = Variable(torch.from_numpy(x)).to(self.device)
    cp_real = []
    for i in components:
      cp_real.append(Variable(torch.from_numpy(i)).to(self.device))

    for i in range(self.num_networks):
      # train discriminator
      cp_fake, mu, logvar = self.generator[i](x_real)
      # l_gan
      y_real_loss = bce_loss(self.discriminator[i](cp_real[i]), t_real)
      y_fake_loss = bce_loss(self.discriminator[i](cp_fake), t_fake)
      L_gan_real = (y_real_loss + y_fake_loss) / 2.0
      # dis_loss
      dis_loss = L_gan_real
      dis_loss.backward()
      self.dis_optim[i].step()
      dis_losses += dis_loss

      # train encoder
      cp_fake, mu, logvar = self.generator[i](x_real)
      # l_prior
      l_prior = 1e-2 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
      # l_like
      l_recon = 5 * l1_loss(cp_fake, cp_real[i])
      l_like = l1_loss(self.discriminator[i].feature(cp_fake), self.discriminator[i].feature(cp_real[i]))
      # enc_loss
      enc_loss = l_prior + l_recon + l_like
      enc_loss.backward()
      self.enc_optim[i].step()
      enc_losses += enc_loss

      # train decoder
      cp_fake, mu, logvar = self.generator[i](x_real)
      # l_gan
      y_fake_loss = bce_loss(self.discriminator[i](cp_fake), t_real)
      L_gan_fake = y_fake_loss
      # l_like
      l_recon = 5 * l1_loss(cp_fake, cp_real[i])
      l_like = l1_loss(self.discriminator[i].feature(cp_fake), self.discriminator[i].feature(cp_real[i]))
      # dec_loss
      dec_loss = l_recon + l_like + 1e-1*L_gan_fake
      dec_loss.backward()
      self.dec_optim[i].step()
      dec_losses += dec_loss

    enc_losses /= self.num_networks
    dec_losses /= self.num_networks
    dis_losses /= self.num_networks
    self.history['dis_loss'].append(dis_losses)
    self.history['enc_loss'].append(enc_losses)
    self.history['dec_loss'].append(dec_losses)
    return enc_losses, dec_losses, dis_losses

  def test(self, x):
    # set to eval mode
    for i in range(self.num_networks):
      self.generator[i].eval()
      self.discriminator[i].eval()
    result = []
    x_real = Variable(torch.from_numpy(x)).to(self.device)
    for i in range(self.num_networks):
      result.append(self.generator[i](x_real)[0].cpu().detach().numpy())
    return result

  def save_model(self, path):
    for i in range(self.num_networks):
      # save generator
      gen_path = path + 'generator_' + str(i)
      dis_path = path + 'discriminator_' + str(i)
      self.generator[i].save_model(gen_path)
      self.discriminator[i].save_model(dis_path)

  def load_model(self, path):
    for i in range(self.num_networks):
      # load
      gen_path = path + 'generator_' + str(i)
      dis_path = path + 'discriminator_' + str(i)
      self.generator[i].load_state_dict(torch.load(gen_path))
      self.discriminator[i].load_state_dict(torch.load(dis_path))

