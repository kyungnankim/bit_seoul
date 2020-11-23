#다중분류
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
#머신러닝 import

#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측

#1.데이터
datasets = load_iris()
x, y = load_iris(return_X_y=True) #자동으로 dataset 넣어준다.
# x = datasets.data
# y = datasets.target

#####1-1. 분류
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66,shuffle=True, train_size=0.8)

#1_2. 데이터 전처리 - OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

#####2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# model.add(Dense(100, activation='relu',input_shape=(4,)))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# model.summary()
model = LinearSVC()
#####3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train,y_train,epochs=1000,batch_size=1,verbose=2) 
model.fit(x_train,y_train)
#####4. 평가, 예측
# loss = model.evaluate(x_test, y_test, batch_size=32)
# print('loss : ', loss)
result = model.score(x_test, y_test)
print("score : ", result)
# y_predict = model.predict(x_test)
