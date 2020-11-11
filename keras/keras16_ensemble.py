#앙상블 기억하기
#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y1 = np.array([range(101,201), range(311,411), range(100)])
x1 = np.transpose(x1)
y1 = np.transpose(y1)

x2 = np.array([range(4,104), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y2 = np.array([range(501,601), range(431,531), range(100, 200)])
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(x2)
print(x2.shape) #(100,3) - 3가지특성의 데이터가 100개이다

from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1,shuffle=True, train_size=0.7)

from sklearn.model_selection import train_test_split 
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2,shuffle=True, train_size=0.7)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model #순차적모델
from tensorflow.keras.layers import Dense, Input #가장 기본적인 모델인 Dense 사용

#모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(5, activation='relu')(input1)
dense1_2 = Dense(4, activation='relu')(dense1_1)
dense1_3 = Dense(3, activation='relu')(dense1_2)
output1 = Dense(3)(dense1_3)

#모델2
input2 = Input(shape=(3,))
dense2_1 = Dense(5, activation='relu')(input2)
dense2_2 = Dense(4, activation='relu')(dense2_1)
dense2_3 = Dense(3, activation='relu')(dense2_2)
output2 = Dense(3)(dense2_3)

#############모델 병합 concatenate (소문자 쓰자~!)
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import concatenate,Concatenate -이전버전
# from keras.layers import concatenate,Concatenate - 이전버전
# 답답하고싶으면 이전버전을 써라 
#성능은 같지만, Concatenate concatenate 사용법이 다름
#concatenate 연산 하지 않음

merge1 = Concatenate()([output1,output2])
#merge1 = concatenate([output1, output2])
#merge1 = Concatenate(axis=1)([output1,output2])
#axis=0 행 기준(같은 column,row+), axis=1이면 열 기준(column늘어나기)
#axis=-1 default => axis=1 과 같음

middle1 = Dense(30)(merge1) #변수명은 틀려도 같아도 ㄱㅊ
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)

#############output 모델 구성
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2_1 = Dense(14)(output2)
output2_2 = Dense(11)(output2_1)
output2_3 = Dense(3)(output2_2)

#총 5개의 모델을 합치기
model = Model(inputs=[input1,input2], outputs=[output1,output2_3])

model.summary()

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit([x1_train,x2_train], [y1_train,y2_train],epochs=100,batch_size=8,validation_split=0.25,verbose=1)

#4.평가, 예측
result = model.evaluate([x1_test,x2_test],[y1_test, y2_test],batch_size=8)
print("result : ",result)