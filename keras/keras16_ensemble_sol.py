#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y1 = np.array([range(101,201), range(311,411), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
print(x1.shape)

x2 = np.array([range(4,104), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y2 = np.array([range(501,601), range(431,331), range(100, 200)])

x2 = np.transpose(x2)
y2 = np.transpose(y2)
print(x2.shape)

from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1,shuffle=True, train_size=0.7)

from sklearn.model_selection import train_test_split 
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2,shuffle=True, train_size=0.7)

#2. 모델구성

print(x2)
print(x2.shape) #(100,3) - 3가지특성의 데이터가 100개이다.

from tensorflow.keras.models import Sequential, Model#순차적모델
from tensorflow.keras.layers import Dense, Input #가장 기본적인 모델인 Dense 사용

input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(3)(dense2)

input2 = Input(shape=(3,))
dense4 = Dense(5, activation='relu')(input2)
dense5 = Dense(4, activation='relu')(dense4)
dense6 = Dense(3, activation='relu')(dense5)
output2 = Dense(3)(dense6)
model = Model(inputs=[input1,input2], outputs=[output1,output2])


model.summary()