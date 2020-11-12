#1. 데이터
import numpy as np 
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]]) #(13,3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_input = np.array([50,60,70])
print(x.shape)
x = x.reshape(13,3,1)
x_input = x_input.reshape(1,3,1)
from sklearn.model_selection import train_test_split
# 실습 LSTM 완성하시오
# 예측값 80

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
# model = Sequential()
# model.add(LSTM(30,activation='relu',input_shape=(3,1)))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(7,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(1))
input1 = Input(shape=(3,)) #함수식
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense2)
model = Model(inputs=input1, outputs=output1)

model.summary()
#3. 컴파일, 훈련

model.compile(loss="mse",optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1,verbose=2)

#4. 예측
y_predict = model.predict(x_input)
print(y_predict)