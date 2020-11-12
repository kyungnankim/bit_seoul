import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #Trainingrja data
y_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) 

# x_val = np.array([11,12,13,14,15]) #validation data
# y_val = np.array([11,12,13,14,15])

x_test = np.array([16,17,18,19,20]) #test data
y_test = np.array([16,17,18,19,20]) 

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential() 
model.add(Dense(30, input_dim=1)) 
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, validation_split=0.2) 

#validation_split
#검증 진행 train의 20% catch   ex) 30 中 24 train  / 6 validation
 #validation_data=(x_val, y_val)) model.fit → 모델훈련
# model.fit(x_train, y_train, epochs = 100) model.fit → 모델훈련

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