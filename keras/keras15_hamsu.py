#1. 데이터
import numpy as np
x = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y = np.array(range(101,201))

print(x)
print(x.shape) #(3,100)
print(y.shape)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x)
print(x.shape) #(100,3) - 3가지특성의 데이터가 100개이다.

#2. 모델구성
from tensorflow.keras.models import Sequential, Model#순차적모델
from tensorflow.keras.layers import Dense, Input #가장 기본적인 모델인 Dense 사용

# model = Sequential()
# model.add(Dense(5, input_shape=(3,),activation='relu')) #만약 (100,10,3)의 데이터가 있다면 input_shape=(10,3) 행무시
# model.add(Dense(4,activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1)) #y는 한개의 컬럼이기 때문에 output 1개

input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
#activation = 활성화 함수, layer마다 활성화 함수 존재
# relu = 평균 85점 이상 나옴 
#선형회귀 모델(default→ activation='linear'])
#회귀 = regress, 선형 = linear
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense2)
model = Model(inputs=input1, outputs=output1)

model.summary()
'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.25, verbose=2) 
#verbose 중간에 실행과정을 생략할 수 있다.  0은 훈련과정생략(보여주는시간이 아깝기 때문에 2사용)

#4. 평가, 예측
loss = model.evaluate(x_train, y_train, batch_size=1)
y_pred = model.predict(x_test)

print("y_test : \n", y_test)
print("y_pred : \n", y_pred)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))
 #R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)
'''