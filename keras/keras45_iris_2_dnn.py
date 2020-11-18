#다중분류
import numpy as np
from sklearn.datasets import load_iris

#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x)
print(x.shape, y.shape) #(506, 13) (506,)

#1-1. 분류
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.4)

#1_2. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(100, activation='relu',input_shape=(4,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=5,mode='auto')
model.fit(x,y,epochs=1000,batch_size=1,verbose=2,callbacks=[early_stopping]) 

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_actually =  np.argmax(y_test, axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)
