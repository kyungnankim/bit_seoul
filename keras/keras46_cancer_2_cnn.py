from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset= load_breast_cancer()

x = dataset.data
y = dataset.target
print(x)
print(x.shape, y.shape) #(569, 30) (569,)

#1.1 데이터 자르기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


# #1_2. 데이터 reshape
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)

#.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten
model=Sequential()
model.add(Conv2D(3, (2,2), padding='same' ,input_shape=(30,1,1)))
model.add(Conv2D(15, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(Conv2D(10, (2,2), padding='same'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
print('예측값 : ', y_predict)
