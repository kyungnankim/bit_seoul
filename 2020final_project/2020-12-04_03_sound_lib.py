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


    all_data = np.concatenate([Xtrain, Xtest], axis=0)
print(all_data.shape)
time.sleep(1)
all_dbmel = gen_4_mels(all_data, normalize=True)
Xtrain_dbmel = all_dbmel[:len(Ytrain)]
Xtest_dbmel = all_dbmel[len(Ytrain):]
print(Xtrain_dbmel.shape, Ytrain.shape, Xtest_dbmel.shape)

import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, BatchNormalization, Activation, Flatten, Dropout, Dense, Add, AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.losses import KLDivergence
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam

def mish(x):
    return x * K.tanh(K.softplus(x))

def eval_kldiv(y_true, y_pred):
    return KLDivergence()(np.array(y_true).astype('float32'), np.array(y_pred).astype('float32')).numpy()

def build_fn():
    dropout_rate=0.5
    
    model_in = Input(shape = (Xtrain_dbmel.shape[1:]))
    x = Convolution2D(32, 3, padding='same', kernel_initializer='he_normal')(model_in)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation(mish)(x)
    x = Convolution2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = Convolution2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation(mish)(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation(mish)(x)
    x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation(mish)(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation(mish)(x)
    x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation(mish)(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Convolution2D(64, 1, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(256, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation(mish)(x)
    x = Convolution2D(64, 1, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(256, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = Convolution2D(64, 1, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(256, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation(mish)(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Convolution2D(128, 1, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(512, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation(mish)(x)
    x = Convolution2D(128, 1, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(512, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(mish)(x)
    x = Convolution2D(128, 1, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = Convolution2D(512, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = Activation(mish)(x)
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)


    x = Flatten()(x)

    x = Dense(units=128, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Activation(mish)(x)
    x = Dropout(rate=dropout_rate)(x)

    x = Dense(units=128, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x_res, x])
    x = Activation(mish)(x)
    x = Dropout(rate=dropout_rate)(x)

    model_out = Dense(units=30, activation='softmax')(x)
    model = Model(model_in, model_out)
    model.compile(loss=KLDivergence(), optimizer=Nadam(learning_rate=0.002))
    return model
build_fn().summary()


num_models=15
model_list=[]

for i in tqdm(range(num_models)):
    model = build_fn()
    model.fit(Xtrain_dbmel, Ytrain, epochs=187, batch_size=16)
    model_list.append(model)
    model.save(f"model_{i}.h5")


models = []
for i in tqdm(range(0, 15)):
    model_name = f"model_{i}.h5"
    models.append(keras.models.load_model(model_name, custom_objects={'mish' : mish}))
print(f"{len(models)} models reloaded")

preds = np.zeros(shape=submission.shape)
train_preds = np.zeros(shape = Ytrain.shape)

train_preds_list=[]
test_preds_list=[]
score_list=[]

for model, i in zip(models, range(len(models))):
    a = model.predict(Xtrain_dbmel)
    b = model.predict(Xtest_dbmel)
    eval_score = eval_kldiv(Ytrain, a)
    
    print(f"Model {i+1} Evaluation Score : {eval_score}")
    train_preds = train_preds + a
    preds = preds + b
    
    train_preds_list.append(a)
    test_preds_list.append(b)
    score_list.append(eval_score)
    
train_preds = train_preds / len(models)
preds = preds / len(models)
print(f"\nMean Predictions Evaluation Score : {eval_kldiv(Ytrain, train_preds)}")
simple_average = pd.DataFrame(preds, index=submission.index, columns=submission.columns)
simple_average.to_csv('15 Average Ensemble model.csv')
simple_average.head(10)


