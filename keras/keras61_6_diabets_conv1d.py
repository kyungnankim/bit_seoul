import numpy as np
from sklearn.datasets import load_diabetes

dataset=load_diabetes()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(442, 10) (442,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
model=Sequential()
model.add(Conv1D(200, kernel_size=2, strides=1, padding='same', input_shape=(10, 1)))
model.add(Conv1D(150, kernel_size=2,  padding='same'))
model.add(Conv1D(100, kernel_size=2, padding='same'))
model.add(Conv1D(80, kernel_size=2, padding='same'))
model.add(Conv1D(70, kernel_size=2, padding='same'))
model.add(Conv1D(50, kernel_size=2, padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='loss', patience=10, mode='min')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2 ,callbacks=[early_stopping])

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)
print("R2 : ", r2)

'''
RMSE :  62.12287785040778
R2 :  0.37241401458897316
'''