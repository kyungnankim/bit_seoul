#1. 데이터
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #(4,3)
# y = np.array([[4,5,6,7]])                     #(1,4)
y = np.array([4,5,6,7])                         #(4, )


print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
# x = x.reshape(4, 3, 1)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(20, input_shape=(3, 1), activation='relu'))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

model.summary()
'''
params = dim(W)+dim(V)+dim(U) = n*n + kn + nm
# n - dimension of hidden layer
# k - dimension of output layer
# m - dimension of input layer
한 개씩 작업하기 위해 LSTM 명시
명시적으로는 (3, ) 이지만 LSTM에 명시를 해주고, Shape 에도 명시를 해줘야 한다.
(4,3) = 12, (4,3,1) = 12 둘의 표현은 같다.
  ↓
4,3,1 → [[[1],[2],[3]]]
LSTM :3차원
'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.25)

x_input = np.array([5,6,7]) 
x_input = x_input.reshape(1,3,1)

#4. 평가, 예측

loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
y_pred = model.predict(x_input)

print("y_pred : \n", y_pred)