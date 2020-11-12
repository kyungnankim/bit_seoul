#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
x2 = np.array([range(4,104), range(761,861), range(100)]) #100개의 데이터 3개 - 100행 3열
y1 = np.array([range(101,201), range(311,411), range(100)])
y2 = np.array([range(501,601), range(431,531), range(100, 200)])
y3 = np.array([range(501,601), range(431,531), range(100, 200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape) 
print(y3.shape) #

#2. 모델구성
from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1,shuffle=True, train_size=0.7)

from sklearn.model_selection import train_test_split 
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x2, y2, y3, shuffle=True, train_size=0.7)


from tensorflow.keras.models import Sequential, Model#순차적모델
from tensorflow.keras.layers import Dense, Input #가장 기본적인 모델인 Dense 사용

#모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(5, activation='relu')(input1)
dense1_2 = Dense(4, activation='relu')(dense1_1)
dense1_3 = Dense(3, activation='relu')(dense1_2)
output1 = Dense(3)(dense1_3)

#모델2
input2 = Input(shape=(3,))
dense2_1 = Dense(5, activation='relu')(input2)
dense2_2 = Dense(4, activation='relu')(dense2_1)
dense2_3 = Dense(3, activation='relu')(dense2_2)
output2 = Dense(3)(dense2_3)

from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import concatenate,Concatenate 

merge1 = concatenate([output1, output2])

middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)

output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2_1 = Dense(14)(output2)
output2_2 = Dense(11)(output2_1)
output2_3 = Dense(3)(output2_2)

output3 = Dense(15)(middle1)
output3_1 = Dense(14)(output3)
output3_2 = Dense(11)(output3_1)
output3_3 = Dense(3)(output3_2)

model = Model(inputs=[input1,input2], outputs=[output1,output2_3,output3_3])


model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit([x1_train,x2_train],
        [y1_train,y2_train,y3_train],epochs=100,batch_size=3,validation_split=0.25,verbose=1)
result =model.evaluate([x1_test,x2_test],[y1_test, y2_test,y3_test],batch_size=3)

print("result : ",result)