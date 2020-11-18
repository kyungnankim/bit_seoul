import numpy as np
from sklearn.datasets import load_diabetes
#1.데이터
dataset=load_diabetes()
x=dataset.data
y=dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(10,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='loss', patience=50, mode='min')
model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2 ,callbacks=[es])

#4.평가
loss,mse = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",loss)
print("mse : ",mse)

print("실제값 : ", y_pred)

#RMSE
y_predict =  model.predict(x_test) 
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_predict))
#R2
from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)