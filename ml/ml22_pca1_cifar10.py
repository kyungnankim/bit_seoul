#pca 축소해서 모델을 완성하세요.
#1. 0.95  이상
#2. 1이상
#minist dnn 과 loss/acc 를 비교

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x = np.append(x_train, x_test, axis=0)
print(x.shape) #(60000, 32, 32, 3)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])

#PCA로 컬럼 걸러내기
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
# print(cumsum)

d = np.argmax(cumsum >= 0.95) + 1
print(d) # 217

pca1=PCA(n_components=d)
x=pca1.fit_transform(x)

x_train=x[:50000, :]
x_test=x[50000:, :]

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.



#2. 모델
model=Sequential()
model.add(Dense(300, activation='relu', input_shape=(d,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', patience=50, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=2, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=128)

print('loss : ', loss)
print('accuracy : ', accuracy)

'''
loss :  5.350092887878418
accuracy :  0.48080000281333923   


'''