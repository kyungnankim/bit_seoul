

import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set_style('whitegrid')
import warnings ; warnings.filterwarnings('ignore')
data_dir = './2020final_project/'

def data_loader(files):
    out = []
    for file in tqdm(files):
        data, fs = librosa.load(file, sr = None)
        out.append(data)
    out = np.array(out)
    return out

Xtrain = glob(data_dir + 'bass_electronic_018-022-100.wav')
Xtrain = data_loader(Xtrain)

print(Xtrain.shape)
time.sleep(1)

Xtrain = Xtrain.astype('float32')


print(Xtrain.shape)

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
    # if save == True:
    #      np.save('./2020final_project/data_dir.npy',arr=data_dir)
    return array
np.save('./2020final_project/data_dir.npy',arr=data_dir)
