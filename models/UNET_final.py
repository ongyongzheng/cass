import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        kernel_size = 5
        padding = 2
        # conv2:
        self.conv2 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        )
        # conv3:
        self.conv3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )
        # conv4:
        self.conv4 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        )
        # conv5:
        self.conv5 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
        )
        # conv6:
        self.conv6 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(512),
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
        self.deconv6 = nn.Sequential(
        nn.ConvTranspose2d(64, 1, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
        nn.BatchNorm2d(1),
        nn.ReLU(inplace=True)
        )
         # conv6: 32 x 64 x 64 -> 3 x 64 x 64
        self.deconv7 = nn.Sequential(
        nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
        nn.BatchNorm2d(1),
        nn.Sigmoid()
        )
        # fc_internal to set feature layer
        self.fc_in = nn.Sequential(
          nn.Linear(512 * 16 * 4, 100, bias=False)
        )
        self.fc_out = nn.Sequential(
          nn.Linear(100, 512 * 16 * 4, bias=False)
        )

    def forward(self, x):
        x_in = torch.log1p(x)
        x_in = x_in[:,:-1,:]
        x_in = x_in.view(-1, 1, 512, 128)
        enc2 = self.conv2(x_in)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        fc = enc6.view(-1, 512*16*4)
        fc = self.fc_in(fc)
        fc = self.fc_out(fc)
        enc6 = fc.view(-1, 512, 16, 4)
        x_out = self.deconv2(enc6)
        x_out = torch.cat([x_out, enc5], dim=1)
        x_out = self.deconv3(x_out)
        x_out = torch.cat([x_out, enc4], dim=1)
        x_out = self.deconv4(x_out)
        x_out = torch.cat([x_out, enc3], dim=1)
        x_out = self.deconv5(x_out)
        x_out = torch.cat([x_out, enc2], dim=1)
        mask = self.deconv6(x_out)
        mask = F.pad(mask, (0, 0, 1, 0), value=0.5)
        mask = torch.squeeze(mask)
        return x * mask

    def save_model(self, path):
        torch.save(self.state_dict(), path)

class UNET(object):
  def __init__(self,
      num_networks,
      learning_rate,
      device,
      Nz,
      weight_decay
  ):
    self.num_networks = num_networks
    self.learning_rate = learning_rate
    self.device = device
    self.Nz = Nz
    self.weight_decay = weight_decay

  def build_model(self):
    self.unet = []
    self.u_optim = []

    for i in range(self.num_networks):
      ae = AE().to(self.device)
      self.unet.append(ae)

      # create optimizers
      u_op = optim.Adam(
        self.unet[i].parameters(),
        lr= self.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay)
      self.u_optim.append(u_op)

    # create history data
    self.history = {}
    self.history['u_loss'] = []

  def train(self, x, components):
    # set to train mode
    for i in range(self.num_networks):
      self.unet[i].train()

    # set up loss functions
    def l1_loss(input, target):
      return torch.mean(torch.abs(input - target))
    def l2_loss(input, target):
      return torch.mean((input - target).pow(2))

    u_losses = 0

    batch_size = x.shape[0]
    # reset gradients
    for i in range(self.num_networks):
      self.u_optim[i].zero_grad()

    x_real = Variable(torch.from_numpy(x)).to(self.device)
    cp_real = []
    for i in components:
      cp_real.append(Variable(torch.from_numpy(i)).to(self.device))

    for i in range(self.num_networks):
      # train unet
      cp_fake = self.unet[i](x_real)
      # l_recon
      l_recon = 5 * l1_loss(cp_fake, cp_real[i])
      # enc_loss
      unet_loss = l_recon
      unet_loss.backward()
      self.u_optim[i].step()
      u_losses += unet_loss

    u_losses /= self.num_networks
    self.history['u_loss'].append(u_losses)
    return u_losses

  def test(self, x):
    # set to eval mode
    for i in range(self.num_networks):
      self.unet[i].eval()

    result = []
    x_real = Variable(torch.from_numpy(x)).to(self.device)
    for i in range(self.num_networks):
      result.append(self.unet[i](x_real).cpu().detach().numpy())
    return result

  def save_model(self, path):
    for i in range(self.num_networks):
      # save vae
      u_path = path + 'unet_' + str(i)
      self.unet[i].save_model(u_path)

  def load_model(self, path):
    for i in range(self.num_networks):
      # load
      u_path = path + 'unet_' + str(i)
      self.unet[i].load_state_dict(torch.load(u_path))

