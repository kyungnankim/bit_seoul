# https://dacon.io/competitions/official/235616/codeshare/1571
import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
import librosa
data_dir='./2020fidown/'
data_pro='./2020final_project/'
nsynth_test = 'nsynth-test/audio/'
nsynth_train = 'nsynth-train/audio/'
nsynth_valid = 'nsynth-valid/audio/'

def data_loader(files):
    out = []
    for file in tqdm(files):
        data, fs = librosa.load(file, sr = None)
        out.append(data)
    out = np.array(out)
    return out

Xtrain = glob(data_dir + 'nsynth-train/audio/*.wav')
Xtrain = data_loader(Xtrain)

# Xtest = glob(data_dir + 'nsynth-test/audio/*.wav')
# Xtest = data_loader(Xtest)
print(Xtrain)
print(Xtrain.shape)
# print(Xtest.shape)
# Xtrain = Xtrain.astype('float32')
# Xtest = Xtest.astype('float32')

# print(Xtrain.shape, Xtest.shape)

# def get_melspectrogram(data, n_fft, win_length, hop_length, n_mels, sr=16000, save=False, to_db=True, normalize=False):
#     array = []
#     for i in tqdm(range(len(data))):
#         melspec = librosa.feature.melspectrogram(data[i], sr=sr, n_fft=n_fft, win_length=win_length, 
#                                                  hop_length=hop_length,n_mels=n_mels)
#         array.append(melspec)
#     array = np.array(array)
#     if to_db == True:
#         array = librosa.power_to_db(array, ref = np.max)
#     if normalize==True: 
#         mean = array.mean()
#         std = array.std()
#         array = (array - mean) / std
#     if save == True:
#         np.save(data_pro + "{data_dir}mel_spectrogram({n_fft},{win_length},{hop_length},{n_mels}).npy", array) 
#     return array
# def gen_4_mels(data, normalize=True):
#     alpha = get_melspectrogram(data, n_fft=256, win_length=200, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
#     beta = get_melspectrogram(data, n_fft=512, win_length=400, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
#     gamma = get_melspectrogram(data, n_fft=1024, win_length=800, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
#     delta = get_melspectrogram(data, n_fft=2048, win_length=1600, hop_length=160, n_mels=64, save=False, to_db=True, normalize=normalize)
    
#     data = np.stack([alpha, beta, gamma, delta], axis=-1)
#     return data
# all_data = np.concatenate([Xtrain, Xtest], axis=0)
# print(all_data.shape)

# import keras
# import keras.backend as K
# from keras.models import Model, Sequential
# from keras.layers import Input, Convolution2D, BatchNormalization, Activation, Flatten, Dropout, Dense, Add, AveragePooling2D
# from keras.callbacks import EarlyStopping
# from keras.losses import KLDivergence
# from sklearn.model_selection import train_test_split
# from keras.optimizers import Nadam

# def mish(x):
#     return x * K.tanh(K.softplus(x))

# def eval_kldiv(y_true, y_pred):
#     return KLDivergence()(np.array(y_true).astype('float32'), np.array(y_pred).astype('float32')).numpy()