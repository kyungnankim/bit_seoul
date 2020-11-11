#1. 데이터
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y = np.array([range(101,201), range(311,411), range(100)])

print(x)
print(x.shape) #(3,100)

# 과제(100,3)으로 바꿔보기
# x = x.T - [[1,711,0], [2,712,1] ...] 형태
# x = x.transpose()
# print(x.shape) 

# x=x.reshape(100,3)
# print(x)  # [[1,2,3], [4,5,6] ...] 형태
# print(x.shape)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split 
x_org, x_test, y_org, y_test = train_test_split(x, y, train_size=0.7)
x_train, x_val, y_train, y_val = train_test_split(x_org, y_org, train_size=0.6)

print(x)
print(x.shape) #(100,3) - 3가지특성의 데이터가 100개

#y1, y2, y3 = w1x1 + w2x2 + w3x3 + b

#2. 모델구성
from tensorflow.keras.models import Sequential #순차적모델
from tensorflow.keras.layers import Dense #가장 기본적인 모델인 Dense 사용

model = Sequential()
model.add(Dense(10, input_dim=3)) #3가지 input
model.add(Dense(5))
model.add(Dense(3)) #따라서 3개의 output 나와야 함

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_data=(x_val, y_val))

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