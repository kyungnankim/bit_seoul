import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.datasets import mnist,cifar10,fashion_mnist
#1.데이터

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()
x_predict=x_test[:10, :, :]
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# print(y_train.shape, y_test.shape) #(60000,10), (10000,10)
# print(y_train[0])

#MinMaxScaler의 효과를 주는 형변환
x_train=x_train.reshape(60000,28*7, 4).astype('float32')/255. #(60000, 28x28)
x_test=x_test.reshape(10000,28*7, 4).astype('float32')/255.
x_predict=x_predict.reshape(10, 28*7, 4).astype('float32')/255.

#conv1D 모델 구성하시오.
model = Sequential()
model.add(Conv1D(120, input_shape=(28*7, 4), kernel_size=2, strides =1, padding = 'same'))
model.add(Conv1D(150,kernel_size=2, strides =2, padding = 'same'))
model.add(Conv1D(90, kernel_size=2, strides =2, padding = 'same'))
model.add(Dense(60))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))
model.summary()

#3.컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=10, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=500, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1000)

print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict=model.predict(x_predict)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test[:10, :], axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)
