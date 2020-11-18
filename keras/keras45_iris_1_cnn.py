# 다중분류
import numpy as np
from sklearn.datasets import load_iris

#1.데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

#1.1 데이터 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

#1_2. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
#from sklearn.preprocessing import OneHotEncoder #sklearn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#1_3 데이터 스케일러
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#1.3 데이터 reshape
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1], 1, 1)

#.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten
model=Sequential()
model.add(Conv2D(7, (2,2), padding='same' ,input_shape=(4, 1, 1)))
model.add(Conv2D(15, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(30, (2,2), padding='same'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])
#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_actually =  np.argmax(y_test, axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)
