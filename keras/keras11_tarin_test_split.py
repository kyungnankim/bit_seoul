from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

#train_test_split
from sklearn.model_selection import train_test_split

#train_size를 70 test는 30
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7) 
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7 , shuffle=false)
# shuffle = default true, DWTShuffle false, DNotWrite true

print(x_test)
# slicing과 다르게 순차적인 데이터값이 아닌 고른 데이터값이 출력
# 순차적인 data로 training한다면 weight값 제한
# 범위를 크게잡아 골고루 data training 시켜 범위를 일정하게 한 다음
# 범위 안에 나머지 30%에 대해 평가

#2. 모델구성
model = Sequential() 
# model.add(Dense(100, input_dim=1)) 
model.add(Dense(30, input_dim=1))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, validation_split=0.2) 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
print("result : \n", y_pred)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

#R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2: ", r2)