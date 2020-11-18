# 다중분류
import numpy as np
from sklearn.datasets import load_iris

dataset=load_iris()
x=dataset.data
y=dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(3, (1,1), padding='same', input_shape=(4,1,1)))
model.add(Conv2D(10, (1,1), padding='same'))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(Conv2D(30, (2,2), padding='same', strides=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.save("./save/keras45_cnn.h5")

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es=EarlyStopping(monitor='loss', patience=50, mode='auto')
# to_hist=TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
#모델 체크포인트 저장 경로 설정 : epoch의 두자릿수 정수 - val_loss 소수점 아래 넷째자리
modelpath='./model/iris-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist=model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es, cp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test, axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)


##########loss와 acc 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) 

plt.subplot(2,1,1) #2행 1열 중 첫번째

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(loc='upper right')


plt.subplot(2,1,2) #2행 1열 중 두번째

plt.plot(hist.history['accuracy'], marker='.', c='red')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['accuracy', 'val_accuracy'])

plt.show()