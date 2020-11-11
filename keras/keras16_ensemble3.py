#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(711,811), range(100)]) 
x2 = np.array([range(4,104), range(711,811), range(100)]) 
y1 = np.array([range(101,201), range(311,411), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

print(x1.shape)
print(x2.shape)
print(y1.shape)

#2. 모델구성
from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=True, train_size=0.7)

from sklearn.model_selection import train_test_split 
x2_train, x2_test, y1_train, y1_test = train_test_split(x2, y1, shuffle=True, train_size=0.7)

from tensorflow.keras.models import Sequential, Model#순차적모델
from tensorflow.keras.layers import Dense, Input #가장 기본적인 모델인 Dense 사용
#모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(50, activation='relu')(input1)
dense1_2 = Dense(40, activation='relu')(dense1_1)
dense1_2 = Dense(30, activation='relu')(dense1_1)
dense1_2 = Dense(20, activation='relu')(dense1_1)
dense1_2 = Dense(30, activation='relu')(dense1_1)
dense1_2 = Dense(40, activation='relu')(dense1_1)
dense1_3 = Dense(20, activation='relu')(dense1_2)
output1 = Dense(3)(dense1_3)
#모델2
input2 = Input(shape=(3,))
dense2_1 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(40, activation='relu')(dense2_1)
dense2_3 = Dense(30, activation='relu')(dense2_2)
output2 = Dense(3)(dense2_3)

from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2])

middle1 = Dense(30)(merge1)
middle1 = Dense(70)(middle1)
middle1 = Dense(60)(middle1)
middle1 = Dense(50)(middle1)
middle1 = Dense(40)(middle1)
middle1 = Dense(30)(middle1)
middle1 = Dense(11)(middle1)

output1 = Dense(30)(middle1)
output1 = Dense(60)(output1)
output1 = Dense(50)(output1)
output1 = Dense(40)(output1)
output1 = Dense(30)(output1)
output1 = Dense(20)(output1)
output1 = Dense(3)(output1)

model = Model(inputs=[input1,input2], outputs = output1)


model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit([x1_train,x2_train], [y1_train],epochs=32,batch_size=3,validation_split=0.25,verbose=1)
result = model.evaluate([x1_test,x2_test],[y1_test],batch_size=3)

print("result : ", result)


#4. 평가, 예측

y1_pred = model.predict([x1_test,x2_test])
print("y1_pred : \n", y1_pred)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y1_pred):
    return np.sqrt(mean_squared_error(y1_test, y1_pred))
print("RMSE : ", RMSE(y1_test, y1_pred))

#R2
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_pred)
print("R2 : ", r2)