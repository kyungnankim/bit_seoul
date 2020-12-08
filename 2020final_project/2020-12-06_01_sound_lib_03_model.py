import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
import librosa
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
data_dir='./2020final_project/'
data_pro='./2020fidown/'
# def data_loader(files):
#     out = []
#     for file in tqdm(files):
#         data, fs = librosa.load(file, sr = None)
#         out.append(data)
#     out = np.array(out)
#     return out
from xgboost import XGBClassifier
xtrain = np.load(data_dir+'/2020-12--06-01_bass_acoustic_000-030-127.npy', allow_pickle=True)
print(xtrain)
print(xtrain.shape)

x_train,  y_train = train_test_split(xtrain,train_size=0.8, random_state=66,shuffle=True)

model = XGBClassifier()

model.fit(x_train,y_train)

print(model.feature_importances_)



# 시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance_cancer(model):
    n_features = xtrain.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importance_cancer(model)
plt.show()














# Xtest = np.load(data_dir+'/string_acoustic_012-030-127.npy', allow_pickle=True).astype('float32')
# print(Xtest)
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
    
#     model_in = Input(shape = (xtrain.shape[1:]))
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
