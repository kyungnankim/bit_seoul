from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# #전처리 이전에 데이터 저장
# np.save('./data/mnist_x_train.npy', arr=x_train)
# np.save('./data/mnist_x_test.npy', arr=x_test)
# np.save('./data/mnist_y_train.npy', arr=y_train)
# np.save('./data/mnist_y_test.npy', arr=y_test)


#저장한 데이터 불러오기
x_train=np.load('./data/cifar10_x_train.npy')
x_test=np.load('./data/cifar10_x_test.npy')
y_train=np.load('./data/cifar10_y_train.npy')
y_test=np.load('./data/cifar10_y_test.npy')

x_predict=x_test[:10, :, :]

x_train = x_train.astype('float32')/255. #CNN은 4차원이기 때문에 4차원으로 변환, astype -0 형변환
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255

#데이터 전처리
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/cifar10/cifar10_CNN_model.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)


############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/cipar10/cipar10_CNN-03-0.9847.hdf5')

#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Conv2D(3, (2,2), input_shape=(32,32,3)))
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
model3.load_weights('./save/mnist/mnist_CNN_model_weight.h5')

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=32)


print("모델 저장 loss : ", result1[0])
print("모델 저장 accuracy : ", result1[1])

print("가중치 저장 loss : ", result3[0])
print("가중치 저장 accuracy : ", result3[1])

print("체크포인트 loss : ", result2[0])
print("체크포인트 accuracy : ", result2[1])
