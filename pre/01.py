
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential() # 순차적
model.add(Dense(5, input_dim=1, activation='relu')) # 한 개 입력 
model.add(Dense(3)) 
model.add(Dense(1, activation='relu'))

model.summary()