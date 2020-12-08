#https://dacon.io/competitions/official/235616/codeshare/1268
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import keras
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import librosa

# wav 파일로부터 데이터를 불러오는 함수, 파일 경로를 리스트 형태로 입력
def data_loader(files):
    out = []
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)    
    out = np.array(out)
    return out
'''
data_dir='./2020final_project/'
samplerate, data = sio.wavfile.read(data_pro + 'ptr/bass_electronic_001-031-127.wav')
'''
data_pro='./2020fidown/'
# Wav 파일로부터 Feature를 만듭니다.
x_data = glob(data_pro + 'ptr/bass_electronic_001-022-127.wav')
x_data = data_loader(x_data)
x_data = x_data[:, ::8] # 매 8번째 데이터만 사용
x_data = x_data / 30000 # 최대값 30,000 을 나누어 데이터 정규화
x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1) # CNN 모델에 넣기 위한 데이터 shape 변경

# 측정 값
y_data =glob(data_pro + 'ptr/bass_synthetic_023-022-127.wav')
y_data =data_loader(y_data)

# Feature, Label Shape을 확인합니다.
print(x_data.shape, y_data.shape)

# 모델을 만듭니다.
# model = Sequential()
# model.add(Conv1D(16, 32, activation='relu', input_shape=(x_data.shape[1], x_data.shape[2])))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Conv1D(16, 32, activation='relu'))
# model.add(MaxPooling1D())
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(30, activation='softmax'))
# model.compile(loss=tf.keras.losses.KLDivergence(), optimizer='adam')
# model.summary()
from xgboost import XGBClassifier
model = XGBClassifier()

model.fit(x_data,y_data)

print(model.feature_importances_)

# 모델 폴더를 생성합니다.
# model_path = './2020fidown/ptr/'
# if not os.path.exists(model_path):
#   os.mkdir(model_path)

# # Validation 점수가 가장 좋은 모델만 저장합니다.
# model_file_path = model_path + 'Epoch_{epoch:03d}_Val_{val_loss:.3f}.hdf5'
# checkpoint = ModelCheckpoint(filepath=model_file_path, monitor='val_loss', verbose=1, save_best_only=True)

# # 10회 간 Validation 점수가 좋아지지 않으면 중지합니다.
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# # 모델을 학습시킵니다.
# history = model.fit(
#     x_data, y_data, 
#     epochs=100, batch_size=256, validation_split=0.8, shuffle=True,
#     callbacks=[checkpoint, early_stopping]
# )

# 훈련 결과를 확인합니다.
# plt.plot(history.epoch, history.history['loss'], '-o', label='training_loss')
# plt.plot(history.epoch, history.history['val_loss'], '-o', label='validation_loss')
# plt.legend()
# plt.xlim(left=0)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# # 검증 wav 파일로부터 Feature를 만듭니다.
# x_test = glob('data/test/*.wav')
# x_test = data_loader(x_test)
# x_test = x_test / 30000
# x_test = x_test[:, ::8]
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# # 가장 좋은 모델의 weight를 불러옵니다.
# weigth_file = glob('model/*.hdf5')[-1]
# print(weigth_file)
# model.load_weights(weigth_file)

# # 예측 수행
# y_pred = model.predict(x_test)
# print(y_pred)