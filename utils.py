import numpy as np
import librosa

def reverse_stft(imag, phase): # compute reverse stft using magnitude imag and phase phase
    magnitude = imag
    nsample = magnitude * np.exp(1j*phase)
    nsample = np.array(librosa.core.istft(nsample, hop_length=768))
    return nsample

def compute_lx_error(x, y, phase, nm=2): # compute lx norm error between x and y magnitudes with phase
    result = []
    for i in range(x.shape[0]):
        p_sample = phase[i]
        x_sample = reverse_stft(x[i],p_sample)
        y_sample = reverse_stft(y[i],p_sample)
        result.append(
            np.linalg.norm(y_sample-x_sample, ord=nm))
    return np.nanmean(np.array(result))

def compute_sdr(x, y, phase, mag):
    # compute signal distortion ratio using 20*log10(sqrt(mean(square(target)))/sqrt(mean(square(signal-target))))
    result = []
    for i in range(x.shape[0]):
        p_sample = phase[i]
        x_sample = reverse_stft(x[i],p_sample)
        y_sample = reverse_stft(y[i],p_sample)
        noise_j = np.sqrt(np.mean(np.square(x_sample-y_sample)))+1e-16
        y_j = np.sqrt(np.mean(np.square(y_sample)))
        if y_j/noise_j != 0:
            result.append(20*np.log10(y_j/noise_j))
    return np.nanmean(result), np.nanstd(result), np.nanmedian(result), np.nanmin(result), np.nanmax(result)