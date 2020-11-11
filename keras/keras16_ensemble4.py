#1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(711,811), range(100)]) 
y1 = np.array([range(101,201), range(311,411), range(100)])
y2 = np.array([range(501,601), range(431,531), range(100, 200)])
y3 = np.array([range(501,601), range(431,531), range(100, 200)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

print(x1.shape)
print(y1.shape)
print(y2.shape) 
print(y3.shape) 

#2. 모델구성
from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=True, train_size=0.7)

from sklearn.model_selection import train_test_split 
y2_train, y2_test, y3_train, y3_test = train_test_split( y2,y3, shuffle=True, train_size=0.7)

from tensorflow.keras.models import Sequential, Model#순차적모델
from tensorflow.keras.layers import Dense, Input #가장 기본적인 모델인 Dense 사용

#모델1
input1 = Input(shape=(3,))
dense1_1 = Dense(5, activation='relu')(input1)
dense1_2 = Dense(4, activation='relu')(dense1_1)
dense1_3 = Dense(3, activation='relu')(dense1_2)
output1 = Dense(3)(dense1_3)

merge1 = output1

middle1 = Dense(1)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(3)(middle1)

output1 = Dense(1)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(1)(middle1)
output2 = Dense(14)(output2)
output2 = Dense(11)(output2)
output2 = Dense(3)(output2)

output3 = Dense(1)(middle1)
output3 = Dense(14)(output3)
output3 = Dense(11)(output3)
output3 = Dense(3)(output3)

model = Model(inputs=input1, outputs=[output1,output2,output3])

model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.fit(x1_train,[y1_train,y2_train,y3_train],epochs=100,batch_size=3,validation_split=0.25,verbose=1)
result =model.evaluate(x1_test, [y1_test, y2_test,y3_test],batch_size=3)

print("result : ",result)
#------------------------------통과 //밑 에러(해결)

#4. 평가, 예측
y1_pred, y2_pred, y3_pred = model.predict(x1_test)
print("y1_pred : \n", y1_pred, y2_pred, y3_pred)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y1_test,y_pred):
    return np.sqrt(mean_squared_error(y1_test,y_pred))

rmse1 = RMSE(y1_test,y1_pred)
rmse2 = RMSE(y2_test,y2_pred)
rmse3 = RMSE(y3_test,y3_pred)
print("RMSE (y1) : ",rmse1)
print("RMSE (y2) : ",rmse2)
print("RMSE (y3) : ",rmse3)
print("Average RMSE",(rmse1+rmse2+rmse3)/3)

#R2
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test,y1_pred)
r2_2 = r2_score(y2_test,y2_pred)
r2_3 = r2_score(y3_test,y3_pred)
print("r2_1: ",r2_1, "r2_2: ",r2_2,"r2_3: ", r2_3)
print("Average r2",(r2_1+r2_2+r2_3)/3)
