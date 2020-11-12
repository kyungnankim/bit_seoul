#1. 데이터
import numpy as np 
from numpy import array
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]]) #(13,3)
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,70], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110], [100,110,120], 
              [2,3,4], [3,4,5], [4,5,6]]) #(13,3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)
# 실습 앙상블 모델을 만드시오
# 예측값 80

#2. 모델구성
from tensorflow.keras.models import Sequential, Model#순차적모델
from tensorflow.keras.layers import Dense, Input, LSTM #가장 기본적인 모델인 Dense 사용

#모델1
input1 = Input(shape=(3,1))
lstm1 = LSTM(30,activation='relu')(input1)
dense1_1 = Dense(50, activation='relu')(lstm1)
dense1_2 = Dense(40, activation='relu')(dense1_1)
dense1_3 = Dense(5, activation='relu')(dense1_2)
output1 = Dense(3)(dense1_3)

input2 = Input(shape=(3,1))
lstm2 = LSTM(30,activation='relu')(input2)
dense2_1 = Dense(50, activation='relu')(lstm2)
dense2_2 = Dense(40, activation='relu')(dense2_1)
dense2_3 = Dense(5, activation='relu')(dense2_2)
output2 = Dense(3)(dense2_3)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])

middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(3)(middle1)

output1 = Dense(30)(middle1)
output1 = Dense(13)(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1,input2], outputs=output1)

#3. 컴파일
model.compile(loss='mse',optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=1, mode='min')
model.fit([x1,x2],y,epochs=100,batch_size=1,verbose=2,callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate([x1,x2],y,batch_size=1)
print("loss : ",loss)
result1 = model.predict([x1_predict, x2_predict])
result2 = model.predict([x2_predict, x1_predict])
print("result1 : ",result1)
print("result2 : ",result2)
print((result1+result2)/2)
