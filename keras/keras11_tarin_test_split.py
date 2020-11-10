#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split #이름만 봐도 갈라질 것만 같은 느낌
x_train, x_test, y_train, y_Test = train_test_split(x, y, train_size = 0.7,shuffle=False)

print(x_test)
x_train = x[:60]     #60개
y_train = y[:60]
x_test =  x[80:]    #20개
y_test =  y[80:]

#나머지 코드를 완성하시오.
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(30, input_dim = 1))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam',metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 예측
loss, mse = model.evaluate(x_test,y_test)
print("mse : ",mse)

y_predict = model.predict(x_test)
print("결과물 \n: ",y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)