import os
import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
import scipy.fftpack
import hyperparameters as hp

F = hp.F # number of LFCC features
T = hp.T # number of frames


def read_flac_file(flac_file_path):
    signal, sample_rate = sf.read(flac_file_path)
    return signal, sample_rate

def extract_lfcc(signal, sample_rate, n_lfcc=F, target_frames=T):
    stft = np.abs(librosa.stft(signal, n_fft=2048))
    log_spectrogram = librosa.amplitude_to_db(stft)
    lfcc = scipy.fftpack.dct(log_spectrogram, axis=0, type=2, norm='ortho')[:n_lfcc]
    if lfcc.shape[1] > target_frames:
        lfcc = lfcc[:, :target_frames]
    else:
        lfcc = np.pad(lfcc, ((0, 0), (0, target_frames - lfcc.shape[1])), mode='constant')
    return lfcc.T





