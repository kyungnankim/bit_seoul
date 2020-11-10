import numpy as np
#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1,activation='relu'))
model.add(Dense(5))
model.add(Dense(1, activation='relu'))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1)
#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("acc : ",acc)