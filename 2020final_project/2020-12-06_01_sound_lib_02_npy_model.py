import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
data_dir='./2020final_project/'
data_pro='./2020fidown/'


import scipy.signal as signal
import math

import numpy as np

def wav_fft(file_name):
    print("fft start")
    audio_sample, sampling_rate = librosa.load(file_name, sr = None)        
    fft_result = librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann).T                    
    mag, phase = librosa.magphase(fft_result)                    
    print("fft end")
    return mag

#normalize_function
# min_level_db = -100
# def _normalize(S):
#     return np.clip((S-min_level_db)/(-min_level_db), 0, 1)


# y, sr = wav_fft(data_pro + 'nsynth-train/audio/bass_acoustic_000-030-127.wav')
# mag_db = librosa.amplitude_to_db(y)
# mag_n = _normalize(mag_db)

# file_path = wav_fft(data_pro + 'nsynth-train/audio/bass_acoustic_000-030-127.wav')
y, sr = librosa.load(data_pro + 'nsynth-train/audio/bass_acoustic_000-030-127.wav')

fft = np.fft.fft(y)
magnitude = np.abs(fft)
f = np.linspace(0, sr, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]


np.save('./2020final_project/2020-12--06-01_bass_acoustic_000-030-127.npy', arr=left_f)

# def data_loader(files):
#     out = []
#     for file in tqdm(files):
#         data, fs = librosa.load(file, sr = None)
#         out.append(data)
#     out = np.array(out)
#     return out

# audio_path = librosa.load(data_pro + 'nsynth-train/audio/bass_acoustic_000-030-127.wav')
# y, sr = librosa.load(audio_path)
# stft_result = librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=1024)
# D = np.abs(stft_result)
# S_dB = librosa.power_to_db(D, ref=np.max)
# librosa.display.specshow(S_dB, sr=sr, hop_length = 1024, y_axis='linear', x_axis='time', cmap = cm.jet)
# plt.colorbar(format='%2.0f dB')
# plt.show()

'''
간단하게 wav파일을 불러와서 파형을 직접 가공할 수도 있고,
 FFT나 MFCC 등 다양한 형태로 변환하는 기능들도 제공

y: 파형의 amplitude 값
sr: sampling rate
y의 그래프를 그려보면, wav파일에 담긴 파형 자체의 그래프가 나오게 됩니다. 
sr은 오디오 파일에 맞게 설정할 수 있으며, 
설정하지 않으면 기본값인 22050으로 파형을 그리게 됩니다.
빨간 곡선이 있었을 때, 이것을 푸리에 변환을 거치게 되면 
다양한 주파수와 위상을 가지는 sin함수들로 변환시키게 됩니다. 
각 sin함수는 주파수와 위상이 정해져 있고, 
이를 x축이 frequency, y축이 amplitude로 나타낼 수 있습니다. 
핵심은, 어떤 파형이든지 간에, sin함수들로 쪼갤 수 있다는 점입니다. 
수 많은 다양한 sin함수의 조합으로 모든 곡선을 만들 수 있다고 봐도 괜찮을 것 같습니다.

x축은 시간, y축은 Hz, 색상은 dB
win_length는 FFT를 할 때 참조할 그래프의 길이입니다.
hop_length는 얼마만큼 시간 주기를 이동하면서 분석을 할 것인지에 대한 파라미터입니다. 
즉, 칼라맵의 시간 주기라고 볼 수 있습니다.
n_fft는 win_length보다 길 경우 모두 zero padding해서 처리하기 위한 파라미터입니다. 
default는 win_length와 같습니다.
win_length는 FFT를 수행할 시간 간격이고, hop_length는 시간 해상도를 나타내는 값입니다. 이렇게 되면 약간의 overlap이 발생하게 됩니다. hop_length 이후의 시간 단위들에 해당하는 주파수까지 포함하게 된다는 의미입니다. 그럼에도 이렇게 사용하는 건 이점이 있기 때문
n_fft가 win_length보다 클 경우, 큰 구간은 모두 zero padding으로 채우게 됩니다. 이유는 설명드린바와 같이, 주파수 해상도를 높이기 위해서입니다.

사실 설명안한 파라미터 중 window 도 있습니다. STFT 라이브러리에서 default 값은 hann으로 되어 있는데요.
FFT를 할 때 특정 주기 간격으로 한다고 말씀드렸는데, 그 주기가 무한번 반복한다는 가정하에 계산이 이뤄지게 됩니다. 그렇다면 다음과 같이 파형이 생겼을 경우, 연결되는 부분에서 문제가 발생하게 됩니다


'''



# ouwav = librosa.load(data_pro + 'nsynth-test/audio/string_acoustic_012-030-127.wav')

# Xtrain = np.array(inwav[1],dtype=float)
# Xtest = np.array(ouwav[1],dtype=float)
# Xtrain = data_loader(Xtrain)
# Xtest = data_loader(Xtest)

# np.save('./2020final_project/bass_acoustic_000-030-127.npy', arr=Xtrain)
# np.save('./2020final_project/string_acoustic_012-030-127.npy', arr=Xtest)

# Xtrain = Xtrain.astype('float32')
# Xtest = Xtest.astype('float32')

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
#         np.save(f"{data_dir}mel_spectrogram({n_fft},{win_length},{hop_length},{n_mels}).npy", array) 
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
# all_dbmel = gen_4_mels(all_data, normalize=True)


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

# def build_fn():
#     dropout_rate=0.5
    
#     model_in = Input(shape = (all_dbmel.shape[1:]))
#     x = Convolution2D(32, 3, padding='same', kernel_initializer='he_normal')(model_in)
#     x = BatchNormalization()(x)
#     x_res = x
#     x = Activation(mish)(x)
#     x = Convolution2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Activation(mish)(x)
#     x = Convolution2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Add()([x, x_res])
#     x = Activation(mish)(x)
#     x = AveragePooling2D()(x)
#     x = Dropout(rate=dropout_rate)(x)

#     x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x_res = x
#     x = Activation(mish)(x)
#     x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Activation(mish)(x)
#     x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Add()([x, x_res])
#     x = Activation(mish)(x)
#     x = AveragePooling2D()(x)
#     x = Dropout(rate=dropout_rate)(x)

#     x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x_res = x
#     x = Activation(mish)(x)
#     x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Activation(mish)(x)
#     x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Add()([x, x_res])
#     x = Activation(mish)(x)
#     x = AveragePooling2D()(x)
#     x = Dropout(rate=dropout_rate)(x)

#     x = Convolution2D(64, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(256, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x_res = x
#     x = Activation(mish)(x)
#     x = Convolution2D(64, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(256, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Activation(mish)(x)
#     x = Convolution2D(64, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(256, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Add()([x, x_res])
#     x = Activation(mish)(x)
#     x = AveragePooling2D()(x)
#     x = Dropout(rate=dropout_rate)(x)

#     x = Convolution2D(128, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(512, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x_res = x
#     x = Activation(mish)(x)
#     x = Convolution2D(128, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(512, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Activation(mish)(x)
#     x = Convolution2D(128, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = Convolution2D(512, 1, padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Add()([x, x_res])
#     x = Activation(mish)(x)
#     x = AveragePooling2D()(x)
#     x = Dropout(rate=dropout_rate)(x)


#     x = Flatten()(x)

#     x = Dense(units=128, kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x_res = x
#     x = Activation(mish)(x)
#     x = Dropout(rate=dropout_rate)(x)

#     x = Dense(units=128, kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Add()([x_res, x])
#     x = Activation(mish)(x)
#     x = Dropout(rate=dropout_rate)(x)

#     model_out = Dense(units=30, activation='softmax')(x)
#     model = Model(model_in, model_out)
#     model.compile(loss=KLDivergence(), optimizer=Nadam(learning_rate=0.002))
#     return model
# build_fn().summary()
