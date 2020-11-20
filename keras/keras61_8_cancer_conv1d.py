import numpy as np
from sklearn.datasets import load_breast_cancer

dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling1D, Dropout, Conv1D, Flatten
model=Sequential()
model.add(Conv1D(500, kernel_size=2, strides=1, padding='same', input_shape=(30, 1)))
model.add(Conv1D(400, kernel_size=2,  padding='same'))
model.add(Conv1D(300, kernel_size=2, padding='same'))
model.add(Conv1D(200, kernel_size=2, padding='same'))
model.add(Conv1D(100, kernel_size=2, padding='same'))
model.add(Conv1D(50, kernel_size=2, padding='same'))
model.add(Conv1D(20, kernel_size=2, padding='same'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='accuracy', patience=50, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])

#4.
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', accuracy)

