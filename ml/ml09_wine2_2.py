#winequality-white.csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

###### 1. 데이터
#pandas로 csv 불러오기
datasets = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')

print(datasets['quality'].value_counts())

datasets = datasets.to_numpy()

x = datasets[:,:-1]
y = datasets[:,-1]

print(y.shape) 

y = y.reshape(-1,1) 
print(y.shape)

ohe = OneHotEncoder(sparse=False).fit(y)
y = ohe.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
x_train ,x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=66, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

##### 2. 모델 
model = Sequential()
model.add(Dense(128, activation='relu',input_shape=(11,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

###### 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor='val_loss',patience=10,mode='auto')
model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,callbacks=[es],validation_data=(x_val,y_val)) 
model.fit(x_train,y_train)

###### 4. 예측, 평가
loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ",loss)
print("acc : ",acc)

y_predict = model.predict(x_test[:10])
y_predict = np.argmax(y_predict,axis=1) 
y_test_recovery = np.argmax(y_test[:10], axis=1)

print("예측값 : ", y_predict)
print("실제값 : ", y_test_recovery)