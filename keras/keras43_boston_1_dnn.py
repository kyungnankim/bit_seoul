from sklearn.datasets import load_boston
#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)
print(x.shape, y.shape) #(506, 13) (506,)

x_pred = x[:10]
y_pred = y[:10]
#1-1. 분류
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.4)
x_test ,x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.6)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(500, activation='relu',input_shape=(13,)))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=5,mode='auto')
model.fit(x,y,epochs=1000,batch_size=1,verbose=2,callbacks=[es]) 

#4.평가
loss,mse = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",loss)
print("mse : ",mse)

print("실제값 : ", y_pred)
y_predict =  model.predict(x_test) 

#RMSE
import numpy as np
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predicted):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2) 