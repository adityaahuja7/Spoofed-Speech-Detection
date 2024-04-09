import os
import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
import scipy.fftpack
import hyperparameters as hp

F = hp.F  
T = hp.T  

def read_flac_file(flac_file_path):
    signal, sample_rate = sf.read(flac_file_path)
    return signal, sample_rate


def extract_lfcc(signal, sample_rate, n_lfcc= hp.F, target_frames= hp.T):
    stft = np.abs(librosa.stft(signal, n_fft= hp.n_fft, hop_length= hp.hop_length))
    log_spectrogram = librosa.amplitude_to_db(stft)
    lfcc = scipy.fftpack.dct(log_spectrogram, axis=0, type=2, norm='ortho')[:n_lfcc]
    
    # Ensure the LFCC is shaped as (n_lfcc, target_frames)
    if lfcc.shape[1] > target_frames:
        lfcc = lfcc[:, :target_frames]
    else:
        lfcc = np.pad(lfcc, ((0, 0), (0, target_frames - lfcc.shape[1])), 'constant')
    
    # Explicitly reshape to ensure the output is (1, n_lfcc, target_frames)
    # This step is crucial to avoid adding an unintended dimension
    lfcc = lfcc.reshape(1, n_lfcc, target_frames)

    return lfcc
