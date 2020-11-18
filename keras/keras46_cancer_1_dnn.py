import numpy as np
from sklearn.datasets import load_breast_cancer
dataset= load_breast_cancer()

x = dataset.data
y = dataset.target
print(x)
print(x.shape, y.shape) #(569, 30) (569,)

#1.1 데이터 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(200, activation='relu',input_shape=(30,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=5,mode='auto')
model.fit(x,y,epochs=10000,batch_size=1,verbose=2,callbacks=[early_stopping]) 

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
print('예측값 : ', y_predict)

