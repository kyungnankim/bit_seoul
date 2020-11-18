from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_predict = x_test[:10, :, :, :]

x_train = x_train.reshape(50000, 32*32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32*32, 3).astype('float32')/255.
x_predict = x_predict.reshape(10, 32*32, 3).astype('float32')/255.


y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)
#2. 모델
model=Sequential()
model.add(LSTM(100, activation='relu', input_shape=(32*32, 3)))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(100, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=24, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=24)
print('loss : ', loss)
print('acc : ', acc)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)