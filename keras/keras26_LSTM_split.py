import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
 
a = np.array(range(1,11))
 
size = 5
 
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
 
dataset = split_x(a, size)
print('------------------')
print(dataset)
 
x_train = dataset[:, 0:-1]
y_train = dataset[:, -1] # [:, 4]
 
print(x_train.shape) 
print(y_train.shape) 
 
# x_train
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
# y_train
# [ 5  6  7  8  9 10]
 
model = Sequential()
model.add(Dense(33, activation = 'relu', input_shape=(4,)))
model.add(Dense(27))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))
# LSTM을 DNN으로 구현 가능
 
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
 
x_input = np.array([7,8,9,10]) # (4,)
x_input = x_input.reshape(1,4)
 
yhat = model.predict(x_input)
print(yhat)
