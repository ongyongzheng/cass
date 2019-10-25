import os

import librosa
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

from models.UNET_final import UNET
from utils import reverse_stft, compute_lx_error, compute_sdr

print("load data files...")
data_dir = '/data/yongzheng/cass/samples/'
stft_bass = np.float32(np.load(data_dir + 's1_train_mag.npy'))
stft_clar = np.float32(np.load(data_dir + 's2_train_mag.npy'))
stft_mix = stft_bass + stft_clar
#stft_mix = np.float32(np.load(data_dir + 'mix_train_mag.npy'))


stft_bass_test = np.float32(np.load(data_dir + 's1_test_mag.npy'))
stft_clar_test = np.float32(np.load(data_dir + 's2_test_mag.npy'))
stft_mix_test = stft_bass_test + stft_clar_test
#stft_mix_test = np.float32(np.load(data_dir + 'mix_test_mag.npy'))

stft_bass_test_phase = np.float32(np.load(data_dir + 's1_test_pha.npy'))
stft_clar_test_phase = np.float32(np.load(data_dir + 's1_test_pha.npy'))
stft_mix_test_pha = np.float32(np.load(data_dir + 'mix_test_pha.npy'))

stft_bass_train_phase = np.float32(np.load(data_dir + 's1_train_pha.npy'))
stft_clar_train_phase = np.float32(np.load(data_dir + 's2_train_pha.npy'))
stft_mix_train_pha = np.float32(np.load(data_dir + 'mix_train_pha.npy'))
print("data files loaded!")

#### Some configurations
BATCH_SIZE = 4
NUM_NETWORKS = 2
NUM_EPOCHS = 1000
threshold = 20
learning_rate = 0.00002
weight_decay = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = '/data/yongzheng/cass_musdb/results/unet/'

class IndexDataset(Dataset):
    def __init__(self, len):
        self.samples = np.array(range(len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

idx = IndexDataset(stft_mix.shape[0])
dl = DataLoader(idx, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

idx2 = IndexDataset(stft_mix_test.shape[0])
dl2 = DataLoader(idx2, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

unet = UNET(NUM_NETWORKS, learning_rate, device, 100, weight_decay)
unet.build_model()

b_error = []
c_error = []

best_result = 0
counter = 0
curr_epoch = 0
for j in range(1, NUM_EPOCHS+1):
    curr_epoch = j
    for i, batch in enumerate(dl):
        IDX = batch.numpy()
        x = stft_mix[IDX]
        a = stft_bass[IDX]
        b = stft_clar[IDX]

        u_losses = unet.train(x, [a, b])

    b_r = list()
    c_r = list()
    for i, batch in enumerate(dl2):
        IDX = batch.numpy()
        x = stft_mix_test[IDX]
        bass_results, clar_results = unet.test(x)
        b_r.append(bass_results)
        c_r.append(clar_results)
    b_r = np.concatenate(b_r, axis=0)
    c_r = np.concatenate(c_r, axis=0)
    results = [b_r, c_r]

    # compute error
    bass_error = compute_lx_error(stft_bass_test, results[0], stft_mix_test_pha)
    clar_error = compute_lx_error(stft_clar_test, results[1], stft_mix_test_pha)
    bass_sdr, bass_std, bass_med, bass_min, bass_max = compute_sdr(stft_bass_test, results[0], stft_mix_test_pha, stft_mix_test)
    clar_sdr, clar_std, clar_med, clar_min, clar_max = compute_sdr(stft_clar_test, results[1], stft_mix_test_pha, stft_mix_test)
    curr_error = 0.5*(bass_sdr + clar_sdr)
    print(10*"=")
    print("Epoch {}, Bass error: {:.4f}, Clar error: {:.4f}".format(
        j, bass_error, clar_error))
    print("Curr Best: {:.4f}, Avg Error: {:.4f}, Avg sdr: {:.4f}".format(
        best_result, 0.5*(bass_error + clar_error), 0.5*(bass_sdr + clar_sdr)))
    print("     Bass sdr: {:.4f}, Clar sdr: {:.4f}".format(
        bass_sdr, clar_sdr))
    print("STD: Bass sdr: {:.4f}, Clar sdr: {:.4f}".format(
        bass_std, clar_std))
    print("MED: Bass sdr: {:.4f}, Clar sdr: {:.4f}".format(
        bass_med, clar_med))
    print("MIN: Bass sdr: {:.4f}, Clar sdr: {:.4f}".format(
        bass_min, clar_min))
    print("MAX: Bass sdr: {:.4f}, Clar sdr: {:.4f}".format(
        bass_max, clar_max))
    print("AE Loss = {:.4f}".format(u_losses))
    if curr_error > best_result:
        best_result = curr_error
        counter = 0
        # save model
        unet.save_model(save_dir)
        np.save(save_dir + 's1_result', results[0])
        np.save(save_dir + 's2_result', results[1])
    counter += 1

    # save error
    b_error.append(bass_error)
    c_error.append(clar_error)
    np.save(save_dir + 's1_error', np.array(b_error))
    np.save(save_dir + 's2_error', np.array(c_error))
    for key in unet.history.keys():
        np.save(save_dir + key, unet.history[key])
    if counter >= threshold:
        break
print("Training ended at epoch {}, with best testing sdr {:.4f}".format(curr_epoch, best_result))
