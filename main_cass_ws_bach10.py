import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import librosa

from models.CASS_with_share_final import CASS
from utils import reverse_stft, compute_lx_error, compute_sdr

print("load data files...")
data_dir = '/data/yongzheng/cass/samples/'
stft_bass = np.float32(np.load(data_dir + 'bass_train_mag.npy'))
stft_clar = np.float32(np.load(data_dir + 'clar_train_mag.npy'))
stft_sax = np.float32(np.load(data_dir + 'sax_train_mag.npy'))
stft_vio = np.float32(np.load(data_dir + 'vio_train_mag.npy'))
stft_mix = stft_bass + stft_clar + stft_sax + stft_vio
#stft_mix = np.float32(np.load(data_dir + 'mix_train_mag.npy'))


stft_bass_test = np.float32(np.load(data_dir + 'bass_test_mag.npy'))
stft_clar_test = np.float32(np.load(data_dir + 'clar_test_mag.npy'))
stft_sax_test = np.float32(np.load(data_dir + 'sax_test_mag.npy'))
stft_vio_test = np.float32(np.load(data_dir + 'vio_test_mag.npy'))
stft_mix_test = stft_bass_test + stft_clar_test + stft_sax_test + stft_vio_test
#stft_mix_test = np.float32(np.load(data_dir + 'mix_test_mag.npy'))

stft_bass_test_phase = np.float32(np.load(data_dir + 'bass_test_pha.npy'))
stft_clar_test_phase = np.float32(np.load(data_dir + 'clar_test_pha.npy'))
stft_sax_test_phase = np.float32(np.load(data_dir + 'sax_test_pha.npy'))
stft_vio_test_phase = np.float32(np.load(data_dir + 'vio_test_pha.npy'))
stft_mix_test_pha = np.float32(np.load(data_dir + 'mix_test_pha.npy'))

stft_bass_train_phase = np.float32(np.load(data_dir + 'bass_train_pha.npy'))
stft_clar_train_phase = np.float32(np.load(data_dir + 'clar_train_pha.npy'))
stft_sax_train_phase = np.float32(np.load(data_dir + 'sax_train_pha.npy'))
stft_vio_train_phase = np.float32(np.load(data_dir + 'vio_train_pha.npy'))
stft_mix_train_pha = np.float32(np.load(data_dir + 'mix_train_pha.npy'))
print("data files loaded!")

#### Some configurations
BATCH_SIZE = 4
NUM_NETWORKS = 4
NUM_EPOCHS = 1000
learning_rate = 0.00002
weight_decay = 1e-6
threshold = 20
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
save_dir = '/data/yongzheng/cass_final/results/cass_ws/'

class IndexDataset(Dataset):
    def __init__(self, len):
        self.samples = np.array(range(len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

idx = IndexDataset(stft_mix.shape[0])
dl = DataLoader(idx, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

cass = CASS(NUM_NETWORKS, learning_rate, device, 100, weight_decay)
cass.build_model()

b_error = []
s_error = []
c_error = []
v_error = []

best_result = 1
counter = 0
curr_epoch = 0
for j in range(1, NUM_EPOCHS+1):
    curr_epoch = j
    for i, batch in enumerate(dl):
        IDX = batch.numpy()
        x = stft_mix[IDX]
        a = stft_bass[IDX]
        b = stft_sax[IDX]
        c = stft_clar[IDX]
        d = stft_vio[IDX]

        enc_losses, dec_losses, dis_losses = cass.train(x, [a, b, c, d])

    results = cass.test(stft_mix_test)
    # compute error
    bass_error = compute_lx_error(stft_bass_test, results[0], stft_mix_test_pha)
    sax_error = compute_lx_error(stft_sax_test, results[1], stft_mix_test_pha)
    clar_error = compute_lx_error(stft_clar_test, results[2], stft_mix_test_pha)
    vio_error = compute_lx_error(stft_vio_test, results[3], stft_mix_test_pha)
    bass_sdr, bass_std, bass_med, bass_min, bass_max = compute_sdr(stft_bass_test, results[0], stft_mix_test_pha, stft_mix_test)
    sax_sdr, sax_std, sax_med, sax_min, sax_max = compute_sdr(stft_sax_test, results[1], stft_mix_test_pha, stft_mix_test)
    clar_sdr, clar_std, clar_med, clar_min, clar_max = compute_sdr(stft_clar_test, results[2], stft_mix_test_pha, stft_mix_test)
    vio_sdr, vio_std, vio_med, vio_min, vio_max = compute_sdr(stft_vio_test, results[3], stft_mix_test_pha, stft_mix_test)
    curr_error = 0.25*(bass_error + sax_error + clar_error + vio_error)
    print(10*"=")
    print("Epoch {}, Bass error: {:.4f}, Sax error: {:.4f}, Clar error: {:.4f}, Vio error: {:.4f}".format(
        j, bass_error, sax_error, clar_error, vio_error))
    print("Curr Best: {:.4f}, Avg Error: {:.4f}, Avg sdr: {:.4f}".format(
        best_result, curr_error, 0.25*(bass_sdr + sax_sdr + clar_sdr + vio_sdr)))
    print("     Bass sdr: {:.4f}, Sax sdr: {:.4f}, Clar sdr: {:.4f}, Vio sdr: {:.4f}".format(
        bass_sdr, sax_sdr, clar_sdr, vio_sdr))
    print("STD: Bass sdr: {:.4f}, Sax sdr: {:.4f}, Clar sdr: {:.4f}, Vio sdr: {:.4f}".format(
        bass_std, sax_std, clar_std, vio_std))
    print("MED: Bass sdr: {:.4f}, Sax sdr: {:.4f}, Clar sdr: {:.4f}, Vio sdr: {:.4f}".format(
        bass_med, sax_med, clar_med, vio_med))
    print("MIN: Bass sdr: {:.4f}, Sax sdr: {:.4f}, Clar sdr: {:.4f}, Vio sdr: {:.4f}".format(
        bass_min, sax_min, clar_min, vio_min))
    print("MAX: Bass sdr: {:.4f}, Sax sdr: {:.4f}, Clar sdr: {:.4f}, Vio sdr: {:.4f}".format(
        bass_max, sax_max, clar_max, vio_max))
    print("Enc Loss = {:.4f}, Dec Loss = {:.4f}, Dis Loss = {:.4f}".format(enc_losses, dec_losses, dis_losses))
    if curr_error <= best_result:
        best_result = curr_error
        counter = 0
        # save model
        cass.save_model(save_dir)
        np.save(save_dir + 'bass_result', results[0])
        np.save(save_dir + 'sax_result', results[1])
        np.save(save_dir + 'clar_result', results[2])
        np.save(save_dir + 'vio_result', results[3])
    counter += 1

    # save error
    b_error.append(bass_error)
    s_error.append(sax_error)
    c_error.append(clar_error)
    v_error.append(vio_error)
    np.save(save_dir + 'bass_error', np.array(b_error))
    np.save(save_dir + 'sax_error', np.array(s_error))
    np.save(save_dir + 'clar_error', np.array(c_error))
    np.save(save_dir + 'vio_error', np.array(v_error))
    for key in cass.history.keys():
        np.save(save_dir + key, cass.history[key])

    if counter >= threshold:
        break
print("Training ended at epoch {}, with best testing sdr {:.4f}".format(curr_epoch, best_result))
