import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

####1.데이터
(x_train, y_train), (x_test, y_test)=mnist.load_data()
x_predict=x_test[:10, :, :]

#인코딩
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

x_train=x_train.reshape(60000,28,28,1).astype('float32')/255. #픽셀의 최대값은 255이므로
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델 구성
model=load_model('./save/model_test02_2.h5')

#4. 평가, 예측
result=model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', result[0])
print('accuracy : ', result[1])