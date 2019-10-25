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

from models.CASS_final import CASS
from utils import reverse_stft, compute_lx_error, compute_sdr

print("load data files...")
data_dir = '/data/yongzheng/cass/samples/'
stft_vocal = np.float32(np.load(data_dir + 's1_train_mag.npy'))
stft_accom = np.float32(np.load(data_dir + 's2_train_mag.npy'))
stft_mix = stft_vocal + stft_accom


stft_vocal_test = np.float32(np.load(data_dir + 's1_test_mag.npy'))
stft_accom_test = np.float32(np.load(data_dir + 's2_test_mag.npy'))
stft_mix_test = stft_vocal_test + stft_accom_test

stft_vocal_test_phase = np.float32(np.load(data_dir + 's1_test_pha.npy'))
stft_accom_test_phase = np.float32(np.load(data_dir + 's1_test_pha.npy'))
stft_mix_test_pha = np.float32(np.load(data_dir + 'mix_test_pha.npy'))

stft_vocal_train_phase = np.float32(np.load(data_dir + 's1_train_pha.npy'))
stft_accom_train_phase = np.float32(np.load(data_dir + 's2_train_pha.npy'))
stft_mix_train_pha = np.float32(np.load(data_dir + 'mix_train_pha.npy'))
print("data files loaded!")

#### Some configurations
BATCH_SIZE = 4
NUM_NETWORKS = 2
NUM_EPOCHS = 1000
learning_rate = 0.00002
weight_decay = 0
threshold = 20
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
save_dir = '/data/yongzheng/cass_musdb/results/cass/'

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

cass = CASS(NUM_NETWORKS, learning_rate, device, 100, weight_decay)
cass.build_model()

b_error = []
c_error = []

best_result = 100000000
best_sdr = 0
counter = 0
curr_epoch = 0
for j in range(1, NUM_EPOCHS+1):
    curr_epoch = j
    for i, batch in enumerate(dl):
        IDX = batch.numpy()
        x = stft_mix[IDX]
        a = stft_vocal[IDX]
        b = stft_accom[IDX]

        enc_losses, dec_losses, dis_losses = cass.train(x, [a, b])

    b_r = list()
    c_r = list()
    for i, batch in enumerate(dl2):
        IDX = batch.numpy()
        x = stft_mix_test[IDX]
        vocal_results, accom_results = cass.test(x)
        b_r.append(vocal_results)
        c_r.append(accom_results)
    b_r = np.concatenate(b_r, axis=0)
    c_r = np.concatenate(c_r, axis=0)
    results = [b_r, c_r]

    # compute error
    vocal_error = compute_lx_error(stft_vocal_test, results[0], stft_mix_test_pha)
    accom_error = compute_lx_error(stft_accom_test, results[1], stft_mix_test_pha)
    vocal_sdr, vocal_std, vocal_med, vocal_min, vocal_max = compute_sdr(stft_vocal_test, results[0], stft_mix_test_pha, stft_mix_test)
    accom_sdr, accom_std, accom_med, accom_min, accom_max = compute_sdr(stft_accom_test, results[1], stft_mix_test_pha, stft_mix_test)
    curr_error = 0.5*(vocal_error + accom_error)
    curr_sdr = 0.5*(vocal_sdr + accom_sdr)
    print(10*"=")
    print("Epoch {}, vocal error: {:.4f}, accom error: {:.4f}".format(
        j, vocal_error, accom_error))
    print("Curr Best: {:.4f}, Avg Error: {:.4f}, Avg sdr: {:.4f}".format(
        best_result, 0.5*(vocal_error + accom_error), 0.5*(vocal_sdr + accom_sdr)))
    print("     vocal sdr: {:.4f}, accom sdr: {:.4f}".format(
        vocal_sdr, accom_sdr))
    print("STD: vocal sdr: {:.4f}, accom sdr: {:.4f}".format(
        vocal_std, accom_std))
    print("MED: vocal sdr: {:.4f}, accom sdr: {:.4f}".format(
        vocal_med, accom_med))
    print("MIN: vocal sdr: {:.4f}, accom sdr: {:.4f}".format(
        vocal_min, accom_min))
    print("MAX: vocal sdr: {:.4f}, accom sdr: {:.4f}".format(
        vocal_max, accom_max))
    print("Enc Loss = {:.4f}, Dec Loss = {:.4f}, Dis Loss = {:.4f}".format(enc_losses, dec_losses, dis_losses))
    if curr_error <= best_result:
        best_result = curr_error
        counter = 0
        # save model
        #cass.save_model(save_dir)
        np.save(save_dir + 's1_result', results[0])
        np.save(save_dir + 's2_result', results[1])
    if curr_sdr > best_sdr:
        best_sdr = curr_sdr
        counter = 0
        # save model
        cass.save_model(save_dir)
        np.save(save_dir + 's1_result', results[0])
        np.save(save_dir + 's2_result', results[1])
    counter += 1

    # save error
    b_error.append(vocal_error)
    c_error.append(accom_error)
    np.save(save_dir + 's1_error', np.array(b_error))
    np.save(save_dir + 's2_error', np.array(c_error))
    for key in cass.history.keys():
        np.save(save_dir + key, cass.history[key])

    if counter >= threshold:
        break
print("Training ended at epoch {}, with best testing error {:.4f}".format(curr_epoch, best_result))
