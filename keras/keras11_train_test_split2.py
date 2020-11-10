#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split #이름만 봐도 갈라질 것만 같은 느낌
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, shuffle = False)
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test, test_size=0.5, shuffle = False)
print(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
# model.add(Dense(10, input_dim=1))
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 예측
mse = model.evaluate(x_test,y_test, batch_size=1)
print("mse : ",mse)

y_predict = model.predict(x_test)
print("y_predict : ",y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("r2 : ",r2)