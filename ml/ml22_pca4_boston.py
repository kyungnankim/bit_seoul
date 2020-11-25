#pca 축소해서 모델을 완성하세요.
#1. 0.95  이상
#2. 1이상
#minist dnn 과 loss/acc 를 비교

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score
from matplotlib import pyplot as plt

dataset=load_iris()
x=dataset.data
y=dataset.target

pca=PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_)

d=np.argmax(cumsum >= 0.95) + 1
# print(cumsum>=0.95) 
print(d)

pca1=PCA(n_components=d)
x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(d,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(50))
model.add(Dense(3, activation='sigmoid'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(monitor='loss', patience=50, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test, batch_size=1)

print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)
y_actually=np.argmax(y_test, axis=1)
print('실제값 : ', y_actually)
print('예측값 : ', y_predict)
plt.plot([1,2,3], [110,130,120])
plt.show()