
import numpy as np
#OneHotEncoding
#sklearn one hot encoding : 하나의 값만 True, 나머지는 모두 False
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

(x_train, y_train), (x_test, y_test)=mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,) (10000,)

x_predict=x_test[:10, :, :]

# print(x_predict.shape) #(10,28,28)
# print(x_predict)


#1. 데이터
#다중 분류 데이터 전처리 (1).OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# print(y_train.shape, y_test.shape) #(60000,10), (10000,10)
# print(y_train[0])

#MinMaxScaler의 효과를 주는 형변환
x_train=x_train.reshape(60000,28*7, 4).astype('float32')/255. #(60000, 28x28)
x_test=x_test.reshape(10000,28*7, 4).astype('float32')/255.



#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM

model=Sequential()
model.add(LSTM(200, activation='relu', input_shape=(28*7, 4)))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(90))
model.add(Dense(30))
model.add(Dense(10, activation='softmax')) 
#(2). 다중 분류의 output layer의 활성화함수는 softmax를 쓴다.

model.summary()

#3. 컴파일, 훈련
#(3). 다중분류에선 반드시 loss를 categorical_crossentropy로 쓴다. 이걸로 이제 accuracy를 잡아줄 수 있다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# es=EarlyStopping(monitor='loss', patience=50, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=100, batch_size=24, verbose=1, validation_split=0.2)

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=24)

print('loss : ', loss)
print('accuracy : ', accuracy)



x_predict=x_predict.reshape(10, 28*7, 4).astype('float32')/255.
print(x_predict)


y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)

