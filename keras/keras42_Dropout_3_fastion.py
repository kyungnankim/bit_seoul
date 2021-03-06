#OneHotEncodeing

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #dataset인 mnist추가
from tensorflow.keras.datasets import cifar10, fashion_mnist

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

x_predict=x_test[:10, :, :]
print(x_predict.shape) #(10,28,28)

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
#from sklearn.preprocessing import OneHotEncoder #sklearn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #CNN은 4차원이기 때문에 4차원으로 변환, astype -0 형변환
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.
x_predict = x_predict.reshape(10,28,28,1).astype('float32')/255

#2. 모델구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1))) #(28,28,10)
model.add(Dropout(0.2)) #(28,28,10) #100개 중에서 80개만 쓰겠다
model.add(Conv2D(20, (2,2), padding='valid')) #(27,27,20)
model.add(Dropout(0.2)) #(28,28,10)
model.add(Conv2D(30, (3,3))) #(25,25,30)
model.add(Dropout(0.2)) #(28,28,10)
model.add(Conv2D(40, (2,2), strides=2)) #(24,24,40)
model.add(MaxPooling2D(pool_size=2)) #기본 Default는 2이다 - (12,12,40)
model.add(Dropout(0.2)) #(28,28,10)
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2)) #(28,28,10)
model.add(Dense(10)) 
model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #mse - 'mean_squared_error' 가능 
#다중분류에서는 categorical_crossentropy를 사용한다!

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1) #OneHotEncoding -> 디코딩하는 문장, axis=0-열, axis=1-행, 즉 y_pred값 안의 행에서 최대값을 가진 곳의 index값을 추출하겠다
print("y_pred : ", y_pred)
