#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(711,811), range(100)]) #(3,100)
y = np.array([range(101, 201), range(311,411), range(100)])
# (100,3)
#x =x.T
#print(x[10][1])
print(x)
print(x.shape) 

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)  #(100,3)

#y1, y2, y3 = w1x1 + w2x2 + w3x3 + b

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_Test = train_test_split(x, y, train_size = 0.7,shuffle=False)

print(x_test)
x_train = x[:60] 
y_train = y[:60]
x_test =  x[80:]   
y_test =  y[80:]

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(3))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam',metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 예측
loss, mse = model.evaluate(x_test,y_test)
print("mse : ",mse)

y_predict = model.predict(x_test)
print("결과물 \n: ",y_predict)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

 #R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)