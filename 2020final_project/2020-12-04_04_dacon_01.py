import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
import librosa

def data_loader(files):
    out = []
    for file in tqdm(files):
        data, fs = librosa.load(file, sr = None)
        out.append(data)
    out = np.array(out)
    return out

Xtrain = glob(data_dir + 'train/*.wav')
Xtrain = data_loader(Xtrain)

Ytrain = pd.read_csv(data_dir + 'train_answer.csv', index_col='id')
submission = pd.read_csv(data_dir + 'submission.csv', index_col='id')

print(Xtrain.shape, Ytrain.shape)
time.sleep(1)

Xtest = glob(data_dir + 'test/*.wav')
Xtest = data_loader(Xtest)

Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')

print(Xtrain.shape, Ytrain.shape, Xtest.shape, submission.shape)


def get_melspectrogram(data, n_fft, win_length, hop_length, n_mels, sr=16000, save=False, to_db=True, normalize=False):
    array = []
    for i in tqdm(range(len(data))):
        melspec = librosa.feature.melspectrogram(data[i], sr=sr, n_fft=n_fft, win_length=win_length, 
                                                 hop_length=hop_length,n_mels=n_mels)
        array.append(melspec)
    array = np.array(array)
    if to_db == True:
        array = librosa.power_to_db(array, ref = np.max)
    if normalize==True: 
        mean = array.mean()
        std = array.std()
        array = (array - mean) / std
    if save == True:
        np.save(f"{data_dir}mel_spectrogram({n_fft},{win_length},{hop_length},{n_mels}).npy", array) 
    return array

def gen_4_mels(data, normalize=True):
    alpha = get_melspectrogram(data, n_fft=256, win_length=200, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
    beta = get_melspectrogram(data, n_fft=512, win_length=400, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
    gamma = get_melspectrogram(data, n_fft=1024, win_length=800, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
    delta = get_melspectrogram(data, n_fft=2048, win_length=1600, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
    
    data = np.stack([alpha, beta, gamma, delta], axis=-1)
    return data
