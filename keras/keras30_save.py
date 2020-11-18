import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
#2. 모델
model=Sequential()
model.add(LSTM(200, activation='relu', input_shape=(4,1)))
model.add(Dense(180, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))

model.summary()

model.save("./save/keras30.h5") #별도의 경로 없이 저장하면 루트(작업폴더)로 감 
# model.save(".\save\keras28_2.h5") #\n \t 등 주의
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")
