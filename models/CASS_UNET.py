import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from models.baseline import Discriminator

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
        self.fc_in = nn.Sequential(
          nn.Linear(512 * 16 * 4, 100, bias=False)
        )

    def forward(self, x, batch_size=64):
        x = x.view(-1, 1, 512, 128)
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h = self.conv5(h4)
        h = h.view(-1, 512 * 16 * 4)
        h = self.fc_in(h)
        return h, h1, h2, h3, h4

class Decoder(nn.Module):
  def __init__(self, nz):
    super(Decoder, self).__init__()
    kernel_size = 5
    padding = 2

    # input params
    self.nz = nz
    # deconv2:
    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    # deconv3:
    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(512, 128, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    # deconv4:
    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(256, 64, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    # deconv5:
    self.deconv5 = nn.Sequential(
        nn.ConvTranspose2d(128, 32, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
    )
    # conv6:
    self.conv6 = nn.Sequential(
      nn.ConvTranspose2d(64, 1, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
      nn.BatchNorm2d(1),
      nn.Sigmoid()
    )
    self.fc_out = nn.Sequential(
      nn.Linear(100, 512 * 16 * 4, bias=False)
    )

  def forward(self, h, h1, h2, h3, h4):
    h = self.fc_out(h)
    h = h.view(-1, 512, 16, 4)
    x_hat = self.deconv2(h)
    x_hat = torch.cat([x_hat, h4], dim=1)
    x_hat = self.deconv3(x_hat)
    x_hat = torch.cat([x_hat, h3], dim=1)
    x_hat = self.deconv4(x_hat)
    x_hat = torch.cat([x_hat, h2], dim=1)
    x_hat = self.deconv5(x_hat)
    x_hat = torch.cat([x_hat, h1], dim=1)
    mask = self.conv6(x_hat)
    return mask

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()

        # input params
        self.nz = nz

        self.encoder = Encoder(nz)
        self.decoder = Decoder(nz)

    def forward(self, x):
        x_in = torch.log1p(x)
        # remove last frequency bin
        x_in = x_in[:,:-1,:]
        outputs = self.encoder(x_in, x.shape[0])
        mask = self.decoder(*outputs)
        # pad mask
        mask = F.pad(mask, (0, 0, 1, 0), value=0.5)
        mask = torch.squeeze(mask)
        return x * mask

    def save_model(self, path):
        torch.save(self.state_dict(), path)

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
        list(self.generator[i].encoder.parameters())+list(self.generator[i].decoder.parameters()),
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
      cp_fake = self.generator[i](x_real)
      # l_gan
      y_real_loss = bce_loss(self.discriminator[i](cp_real[i]), t_real)
      y_fake_loss = bce_loss(self.discriminator[i](cp_fake), t_fake)
      cross_loss = 0
      for j in range(self.num_networks):
          if j != i:
              cross_loss += bce_loss(self.discriminator[i](self.generator[j](x_real)), t_fake)
      L_gan_real = (y_real_loss + y_fake_loss) / 2.0
      # dis_loss
      dis_loss = L_gan_real + 0.1 * cross_loss
      dis_loss.backward()
      self.dis_optim[i].step()
      dis_losses += dis_loss

      # train encoder
      cp_fake = self.generator[i](x_real)
      # l_like
      l_recon = 5 * l1_loss(cp_fake, cp_real[i])
      l_like = l1_loss(self.discriminator[i].feature(cp_fake), self.discriminator[i].feature(cp_real[i]))
      y_fake_loss = bce_loss(self.discriminator[i](cp_fake), t_real)
      L_gan_fake = y_fake_loss
      # enc_loss
      enc_loss = l_recon + l_like + 1e-1*L_gan_fake
      enc_loss.backward()
      self.enc_optim[i].step()
      enc_losses += enc_loss

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
      result.append(self.generator[i](x_real).cpu().detach().numpy())
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


