#pca 축소해서 모델을 완성하세요.
#1. 0.95  이상
#2. 1이상
#minist dnn 과 loss/acc 를 비교

import numpy as np
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

dataset=load_iris()
x=dataset.data
y=dataset.target
# print(x)
# print(x.shape, y.shape) #(150, 4) (150,)

#PCA로 컬럼 걸러내기
pca=PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시
# print(cumsum)

d=np.argmax(cumsum >= 0.99) + 1
# print(cumsum>=0.95) 
print(d) # 2

pca1=PCA(n_components=d)
x=pca1.fit_transform(x)


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

model=Sequential()
model.add(Dense(300, activation='relu', input_shape=(d,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50))
model.add(Dense(3, activation='softmax'))


model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
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

y = [5, 3, 7, 10, 9, 5, 3.5, 8]
x = range(len(y))
plt.bar(x, y, width=0.7, color="gray")
plt.show()


'''
curacy: 1.0000
loss :  0.0003212341107428074
accuracy :  1.0
실제값 :  [1 0 2 0 2 0 2 1 0 0 2 2 1 2 
1 1 1 2 1 0 0 0 2 0 1 1 1 2 0 1]       
예측값 :  [1 0 2 0 2 0 2 1 0 0 2 2 1 2 
1 1 1 2 1 0 0 0 2 0 1 1 1 2 0 1]   

'''