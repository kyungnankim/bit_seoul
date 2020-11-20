from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_predict=x_test[:10, :, :]


y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)

x_train=x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test=x_test.reshape(10000,28,28,1).astype('float32')/255.
x_predict=x_predict.reshape(10, 28, 28,1).astype('float32')/255.


################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/model_cifar10.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/cipar10_CNN-03-0.9840.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Conv2D(3, (2,2), input_shape=(28,28,1)))
model3.add(Conv2D(10, (2,2)))
model3.add(Conv2D(20, (3,3)))
model3.add(Conv2D(30, (2,2), strides=2))
model3.add(MaxPooling2D())
model3.add(Flatten())
model3.add(Dense(20, activation='relu'))
model3.add(Dense(10, activation='softmax'))

# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cifar10__CNN_model_weight.h5')

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=32)


print("모델 저장 loss : ", result1[0])
print("모델 저장 accuracy : ", result1[1])

print("가중치 저장 loss : ", result3[0])
print("가중치 저장 accuracy : ", result3[1])

print("체크포인트 loss : ", result2[0])
print("체크포인트 accuracy : ", result2[1])
