from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D #CNN 기본 layer
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

print(x_train[0])
print("y_train[0]",y_train[0])
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_predict=x_test[:10, :, :]

#1. 데이터
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
# 레이블 벡터화 
x_train=x_train.reshape(60000,28*28, 1).astype('float32')/255. #(60000, 28x28)
x_test=x_test.reshape(10000,28*28, 1).astype('float32')/255.
x_predict=x_predict.reshape(10, 28*28, 1).astype('float32')/255.

#2. 모델
model=Sequential()
model.add(LSTM(30, activation='relu', input_shape=(28*28, 1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=3, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=24, verbose=1, validation_split=0.2)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=24)

print('loss : ', loss)
print('acc : ', acc)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)

print('실제값 : ', y_actually)
print('예측값 : ', y_predict)