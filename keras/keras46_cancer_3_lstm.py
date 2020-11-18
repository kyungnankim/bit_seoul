from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset= load_breast_cancer()

x = dataset.data
y = dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

#1.1 데이터 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
model=Sequential()
model.add(LSTM(7, activation='relu', input_shape=(30, 1)))
model.add(Dense(40, activation='relu'))
# model.add(Dense(90, activation='relu'))
# model.add(Dense(70, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min')
model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2 ,callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
print('예측값 : ', y_predict)