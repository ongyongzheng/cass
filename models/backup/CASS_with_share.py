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
        nn.Linear(512 * 33 * 2, 1024, bias=False),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True)
        )

        # mu: 1024 -> nz
        self.mu = nn.Linear(1024, nz, bias=False)

        # logvar: 1024 -> nz
        self.logvar = nn.Linear(1024, nz, bias=False)

    def forward(self, x, batch_size=64):
        x = x.view(-1, 1, 513, 22)
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
      nn.Linear(nz, 512 * 33 * 2, bias=False),
      nn.BatchNorm1d(512 * 33 * 2),
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
      nn.ReLU(inplace=True)
    )
    # fc7: 3 x 64 x 64 -> 8514
    self.fc7 = nn.Sequential(
    nn.Linear(1 * 528 * 32, 11286, bias=False),
    nn.Sigmoid()
    )

  def forward(self, z):
    xhat = self.fc1(z)
    xhat = xhat.view(-1, 512, 33, 2)
    xhat = self.deconv2(xhat)
    xhat = self.deconv3(xhat)
    xhat = self.deconv4(xhat)
    xhat = self.deconv5(xhat)
    xhat = self.conv6(xhat)
    xhat = xhat.view(-1, 1 * 528 * 32)
    mask = self.fc7(xhat)
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
        mu, logvar = self.encoder(x_in, x.shape[0])
        z = self.reparameterize(mu, logvar)
        mask = self.decoder(z)
        return x * mask, mu, logvar

    def encode(self, x):
        x_in = torch.log1p(x)
        mu, logvar = self.encoder(x_in)
        return mu

    def generate(self, z):
        self.eval()
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x):
        self.eval()
        x_in = torch.log1p(x)
        mu, logvar = self.encoder(x)
        xhat = self.decoder(mu)

        return x * xhat

    def save_model(self, path):
        torch.save(self.state_dict(), path)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    # fc_in: 3 x 64 x 64
    self.fc_in = nn.Sequential(
    nn.Linear(11286, 3 * 64 * 64, bias=False),
    nn.BatchNorm1d(3 * 64 * 64),
    nn.ReLU(inplace=True)
    )

    # conv1: 3 x 64 x 64 -> 32 x 64 x 64
    self.conv1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.2)
    )

    # conv2: 32 x 64 x 64 -> 64 x 32 x 32
    self.conv2 = nn.Sequential(
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2)
    )

    # conv3: 64 x 32 x 32 -> 128 x 16 x 16
    self.conv3 = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2)
    )

    # conv4: 128 x 16 x 16 -> 256 x 8 x 8
    self.conv4 = nn.Sequential(
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2)
    )

    # conv5: 256 x 8 x 8 -> 512 x 4 x 4
    self.conv5 = nn.Sequential(
    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2)
    )

    # fc6: 512 * 4 * 4 -> 1024
    self.fc6 = nn.Sequential(
    nn.Linear(512 * 33 * 2, 1024, bias=False),
    nn.BatchNorm1d(1024),
    nn.LeakyReLU(0.2)
    )

    # fc7: 1024 -> 1
    self.fc7 = nn.Sequential(
    nn.Linear(1024, 1),
    )


  def forward(self, x):
    #x = self.fc_in(x)
    x = x.view(-1, 1, 513, 22)
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    f = self.conv4(f)
    f = self.conv5(f)
    f = f.view(-1, 512 * 33 * 2)
    f = self.fc6(f)
    o = self.fc7(f)
    return o

  def feature(self, x):
    #x = self.fc_in(x)
    x = x.view(-1, 1, 513, 22)
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    return f.view(-1, 128 * 129 * 6)

  def fc_feature(self, x):
    #x = self.fc_in(x)
    x = x.view(-1, 1, 513, 22)
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    f = self.conv4(f)
    f = self.conv5(f)
    f = f.view(-1, 512 * 33 * 2)
    f = self.fc6(f)
    return f

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
      z_fake_p = Variable(torch.randn(x.shape[0], self.Nz)).to(self.device)
      cp_fake, mu, logvar = self.generator[i](x_real)
      # l_gan
      y_real_loss = bce_loss(self.discriminator[i](cp_real[i]), t_real)
      y_fake_loss = bce_loss(self.discriminator[i](cp_fake), t_fake)
      #y_fake_p_loss = bce_loss(self.discriminator[i](self.generator[i].decoder(z_fake_p) * x_real), t_fake)
      # TODO: Implement loss term for cross term
      cross_loss = 0
      for j in range(self.num_networks):
          if j != i:
              cross_loss += bce_loss(self.discriminator[i](self.generator[j](x_real)[0]), t_fake)
      #L_gan_real = (y_real_loss + y_fake_loss + y_fake_p_loss) / 3.0
      L_gan_real = (y_real_loss + y_fake_loss) / 2.0
      # dis_loss
      dis_loss = L_gan_real + 0.1 * cross_loss
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
      z_fake_p = Variable(torch.randn(x.shape[0], self.Nz)).to(self.device)
      cp_fake, mu, logvar = self.generator[i](x_real)
      # l_gan
      #y_real_loss = bce_loss(self.discriminator[i](cp_real[i]), t_fake)
      y_fake_loss = bce_loss(self.discriminator[i](cp_fake), t_real)
      #y_fake_p_loss = bce_loss(self.discriminator[i](self.generator[i].decoder(z_fake_p) * x_real), t_real)
      #L_gan_fake = (y_fake_loss + y_fake_p_loss) / 2.0
      L_gan_fake = y_fake_loss
      # l_like
      l_recon = 5 * l1_loss(cp_fake, cp_real[i])
      l_like = l1_loss(self.discriminator[i].feature(cp_fake), self.discriminator[i].feature(cp_real[i]))
      # dec_loss
      dec_loss = l_recon + l_like + L_gan_fake
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

