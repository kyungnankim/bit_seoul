import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

#1.데이터
a = np.array(range(1, 100))
size = 5

#split_x 멋진 함수를 데려오고
def split_x(seq, size) :
    aaa=[]
    for i in range(len(seq)-size+1) :
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

datasets=split_x(a, size)
print(datasets)

x = datasets[:, :4]
y = datasets[:, 4]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)
x_predict=np.array([[97, 98, 99, 100]])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_predict=scaler.transform(x_predict)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict=x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)

#conv1D 모델 구성하시오.
model = Sequential()
model.add(Conv1D(120, input_shape=(x_predict.shape[1], 1), kernel_size=2, strides =1, padding = 'same'))
model.add(Dense(150, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(60))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 예측
loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_predict)

print("y_predict :", y_predict)
print("loss : ", loss)
'''
y_predict : [[101.572754]]
loss :  0.11260636150836945
'''