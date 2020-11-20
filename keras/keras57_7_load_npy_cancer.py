# 이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train=np.load('./data/cancer_x_train.npy')
x_test=np.load('./data/cancer_x_test.npy')
y_train=np.load('./data/cancer_y_train.npy')
y_test=np.load('./data/cancer_y_test.npy')


scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

################### 1. load_model ########################

#3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model1 = load_model('./save/cancer/cancer_DNN_model.h5')
#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)



############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model

model2 = load_model('./model/cancer_DNN-54-0.0571.hdf5')
#4. 평가, 예측
result2 = model2.evaluate(x_test, y_test, batch_size=32)


################ 3. load_weights ##################

# 2. 모델
model3 = Sequential()
model3.add(Dense(80, activation='relu', input_shape=(30,)))
model3.add(Dense(350, activation='relu'))
model3.add(Dropout(0.1))
model3.add(Dense(550, activation='relu'))
model3.add(Dense(1000, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(480, activation='relu'))
model3.add(Dense(180, activation='relu'))
model3.add(Dense(80))
model3.add(Dense(1, activation='sigmoid'))

# model.summary()

# 3. 컴파일

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/weight_cancer.h5')

#4. 평가, 예측

result3 = model3.evaluate(x_test, y_test, batch_size=1)


print("모델 저장 loss : ", result1[0])
print("모델 저장 accuracy : ", result1[1])

print("가중치 저장 loss : ", result3[0])
print("가중치 저장 accuracy : ", result3[1])

print("체크포인트 loss : ", result2[0])
print("체크포인트 accuracy : ", result2[1])

