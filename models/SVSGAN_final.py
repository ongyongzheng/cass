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
        self.out = nn.Sequential(
        nn.Linear(1024, nz, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x, batch_size=64):
        x = x.view(-1, 1, 512, 128)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(batch_size, -1)
        h = self.fc6(h)
        out = self.out(h)
        return out

class Decoder(nn.Module):
  def __init__(self, nz, num_networks):
    super(Decoder, self).__init__()
    # input params
    self.nz = nz
    self.num_networks = num_networks
    kernel_size = 5
    padding = 2
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
    self.output = []
    for _ in range(num_networks):
        self.output.append(
            nn.Sequential(
              nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
              nn.Sigmoid()
            )
        )

  def forward(self, z):
    xhat = self.fc1(z)
    xhat = xhat.view(-1, 512, 16, 4)
    xhat = self.deconv2(xhat)
    xhat = self.deconv3(xhat)
    xhat = self.deconv4(xhat)
    xhat = self.deconv5(xhat)
    outputs = []
    for i in range(len(self.output)):
        outputs.append(self.output[i](xhat))
    mask = torch.cat(outputs, dim=1)
    return mask

class Generator(nn.Module):
    def __init__(self, nz, num_networks):
        super(Generator, self).__init__()

        # input params
        self.nz = nz
        self.num_networks = num_networks

        self.encoder = Encoder(nz)
        self.decoder = Decoder(nz, num_networks)

    def forward(self, x):
        x_in = torch.log1p(x)
        # remove last frequency bin
        x_in = x_in[:,:-1,:]
        z = self.encoder(x_in, x.shape[0])
        new_x = x.view(-1, 1, 513, 128)
        x_stack = torch.cat([new_x]*self.num_networks, dim=1)
        mask = self.decoder(z)
        mask = F.pad(mask, (0, 0, 1, 0), value=0.5)
        return torch.cat([new_x,x_stack * mask],dim=1)

    def encode(self, x):
        x_in = torch.log1p(x)
        # remove last frequency bin
        x_in = x_in[:,:-1,:]
        out = self.encoder(x_in, x.shape[0])
        return out

    def generate(self, z):
        self.eval()
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x):
        self.eval()
        x_in = torch.log1p(x)
        out = self.encoder(x_in)
        xhat = self.decoder(out)

        return x * xhat

    def save_model(self, path):
        torch.save(self.state_dict(), path)

class Discriminator(nn.Module):
  def __init__(self, num_networks):
    super(Discriminator, self).__init__()
    self.num_networks = num_networks
    kernel_size = 5
    padding = 2
    # conv1:
    self.conv1 = nn.Sequential(
    nn.Conv2d(1+num_networks, 32, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
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
    x = x[:,:,:-1,:]
    x = x.view(-1, 1+self.num_networks, 512, 128)
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
    x = x[:,:,:-1,:]
    x = x.view(-1, 1+self.num_networks, 512, 128)
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    return f.view(-1, 128 * 64 * 16)

  def save_model(self, path):
    torch.save(self.state_dict(), path)

class SVSGAN(object):
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
    self.generator = Generator(self.Nz, self.num_networks).to(self.device)
    for i in self.generator.decoder.output:
      i.to(self.device)
    self.discriminator = Discriminator(self.num_networks).to(self.device)
    self.enc_optim = optim.Adam(
        self.generator.encoder.parameters(),
        lr= self.learning_rate,
        weight_decay=self.weight_decay,
        betas=(0.5, 0.999))
    self.dec_optim = optim.Adam(
        self.generator.decoder.parameters(),
        lr= self.learning_rate,
        weight_decay=self.weight_decay,
        betas=(0.5, 0.999))
    self.dis_optim = optim.Adam(
        self.discriminator.parameters(),
        lr= self.learning_rate,
        weight_decay=self.weight_decay,
        betas=(0.5, 0.999))

    # create history data
    self.history = {}
    self.history['dis_loss'] = []
    self.history['enc_loss'] = []
    self.history['dec_loss'] = []

  def train(self, x, components):
    # set to train mode
    self.generator.train()
    self.discriminator.train()

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
    self.enc_optim.zero_grad()
    self.dec_optim.zero_grad()
    self.dis_optim.zero_grad()

    x_real = Variable(torch.from_numpy(x)).to(self.device)
    x_stack = torch.cat([x_real]*self.num_networks, dim=1).to(self.device)
    cp_real = []
    for i in components:
      cp_real.append(Variable(torch.from_numpy(i)).unsqueeze(1).to(self.device))
    target = torch.cat([x_real.unsqueeze(1)]+cp_real, dim=1).to(self.device)

    # train discriminator
    cp_fake = self.generator(x_real)
    z_fake_p = Variable(torch.randn(x.shape[0], self.Nz)).to(self.device)
    # l_gan
    y_real_loss = bce_loss(self.discriminator(target), t_real)
    y_fake_loss = bce_loss(self.discriminator(cp_fake), t_fake)
    y_fake_p_loss = bce_loss(self.discriminator[i](self.generator[i].decoder(z_fake_p) * x_stack), t_real)
    L_gan_real = (y_real_loss + y_fake_loss + y_fake_p_loss) / 3.0

    # dis_loss
    dis_loss = L_gan_real
    dis_loss.backward()
    self.dis_optim.step()
    dis_losses += dis_loss

    # train encoder
    cp_fake = self.generator(x_real)
    # l_like
    l_recon = 5 * l1_loss(cp_fake, target)
    # enc_loss
    enc_loss = l_recon
    enc_loss.backward()
    self.enc_optim.step()
    enc_losses += enc_loss

    # train decoder
    cp_fake = self.generator(x_real)
    z_fake_p = Variable(torch.randn(x.shape[0], self.Nz)).to(self.device)
    # l_gan
    y_fake_loss = bce_loss(self.discriminator(cp_fake), t_real)
    y_fake_p_loss = bce_loss(self.discriminator[i](self.generator[i].decoder(z_fake_p) * x_stack), t_real)
    L_gan_fake = (y_fake_loss + y_fake_loss) / 2
    # l_like
    l_recon = 5 * l1_loss(cp_fake, target)
    # dec_loss
    dec_loss = l_recon + L_gan_fake
    dec_loss.backward()
    self.dec_optim.step()
    dec_losses += dec_loss

    self.history['dis_loss'].append(dis_losses)
    self.history['enc_loss'].append(enc_losses)
    self.history['dec_loss'].append(dec_losses)
    return enc_losses, dec_losses, dis_losses

  def test(self, x):
    # set to eval mode
    self.generator.eval()
    self.discriminator.eval()
    result = []
    #x_len = x.shape[1]
    x_real = Variable(torch.from_numpy(x)).to(self.device)
    full_result = np.transpose(self.generator(x_real).cpu().detach().numpy(), (1,0,2,3))
    for i in range(self.num_networks):
      result.append(full_result[i+1])
    return result

  def save_model(self, path):
    # save generator
    gen_path = path + 'generator'
    dis_path = path + 'discriminator'
    self.generator.save_model(gen_path)
    self.discriminator.save_model(dis_path)

  def load_model(self, path):
    gen_path = path + 'generator'
    dis_path = path + 'discriminator'
    self.generator.load_state_dict(torch.load(gen_path))
    self.discriminator.load_state_dict(torch.load(dis_path))
