#1~100까지의 데이터를 LSTM으로 훈련 및 예측
#train, test 분리
#ealry_stopping 사용
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터
dataset = np.array(range(1,101))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) 
    # print(type(aaa))
    return np.array(aaa)
datasets=split_x(dataset, size)
x=datasets[:, :size-1]
y=datasets[:, size-1:]

x=x.reshape(x.shape[0], x.shape[1], 1)

#train과 test 데이터로 가르기
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.9)

model=Sequential()
model.add(LSTM(200, activation='relu', input_shape=(4,1)))
model.add(Dense(180, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['acc'])


early_stopping=EarlyStopping(monitor='loss', patience=40, mode='min')
history=model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2,callbacks=[early_stopping])

loss=model.evaluate(x_test, y_test, batch_size=1)

x_predict=np.array([97,98,99,100])
x_predict=x_predict.reshape(1,4,1)

y_predict=model.predict(x_predict)


print("y_predict : ", y_predict)
print("loss : ", loss)
# print("=================================")
# print("history : ", history)
# print("=================================")
# print(history.history.keys())
# print("=================================")
# print(history.history['loss'])
# print("=================================")


# 그래프
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_loss'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()