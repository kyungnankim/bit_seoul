import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_diabetes #load_diabetes 당뇨병데이터

#1. 데이터
# Attribute Information (in order):
#     0    - age     나이
#     1    - sex     성별
#     2    - bmi     bmi 체질량 지수
#     3    - bp      bp 평균 혈압
#     4    - s1 tc   T- 세포 (백혈구의 일종)
#     5    - s2 ldl  저밀도 지단백질
#     6    - s3 hdl  고밀도 지단백질
#     7    - s4 tch  갑상선 자극 호르몬
#     8    - s5 ltg  라모트리진
#     9    - s6 glu   혈당 수치
#    10    - target  1년 후 질병 진행의 측정

dataset = load_diabetes()
x = dataset.data #(442,10), 인덱스 0~9의 값
y = dataset.target #(442,), 인덱스 10의 값- 1년 후 당뇨병 진행의 측정
#x의 데이터로 1년 후 당뇨병 진행을 측정하는 데이터셋이다.

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_standard = scaler.transform(x) 
x_standard = x_standard.reshape(442,5,2,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 


#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (5,2), padding='same', input_shape=(5,2,1))) #(5,2,10)
model.add(Dropout(0.2))
model.add(Conv2D(20, (4,1), padding='valid')) #(2,2,20)
model.add(Dropout(0.2))
model.add(Conv2D(30, (2,2), padding='same')) #(2,2,30)
model.add(Dropout(0.2))
# model.add(Conv2D(40, (2,2), strides=2)) #(2,2,40)
# model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2)) #(1,1,30)
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#print(y_predict.shape) #(89, 1)

print("y_test : ", y_test)
print("y_predict : \n", y_predict.reshape(89,))

from sklearn.metrics import mean_absolute_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)