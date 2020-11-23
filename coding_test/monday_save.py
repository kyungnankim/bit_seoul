import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd

def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)

samsung = pd.read_csv('./data/csv/samsung_1120.csv', header=0, index_col=0, encoding='CP949', sep=',' )
hite = pd.read_csv('./data/csv/bit_1120.csv', header=0, index_col=0, encoding='CP949',sep=',' )
gold = pd.read_csv('./data/csv/금현물.csv', header=0, index_col=0, encoding='CP949', sep=',' )
kosdaq = pd.read_csv('./data/csv/kosdaq.csv', header=0, index_col=0, encoding='CP949',sep=',' )

#정렬을 일자별 오름차순으로 변경
samsung=samsung.sort_values(['일자'], ascending=['True'])
hite=hite.sort_values(['일자'], ascending=['True'])
gold=gold.sort_values(['일자'], ascending=['True'])
kosdaq=kosdaq.sort_values(['일자'], ascending=['True'])

#필요한 컬럼만
samsung=samsung[['시가', '고가', '저가', '개인', '종가']]
hite=hite[['시가', '고가', '저가', '개인', '종가']]
gold=gold[['시가', '고가', '저가', '종가', '거래량', '거래대금(백만)']]
kosdaq=kosdaq[['시가', '저가', '고가']]


#콤마 제거 후 문자를 정수로 변환
for i in range(len(samsung.index)) :
    for j in range(len(samsung.iloc[i])) :
        samsung.iloc[i, j]=int(samsung.iloc[i, j].replace(',', ''))

for i in range(len(hite.index)) :
    for j in range(len(hite.iloc[i])) :
        hite.iloc[i, j]=int(hite.iloc[i, j].replace(',', ''))

for i in range(len(gold.index)) :
    for j in range(len(gold.iloc[i])) :
        gold.iloc[i, j]=int(gold.iloc[i, j].replace(',', ''))

samsung_target=samsung[['고가', '저가', '개인','종가']]
samsung_data=samsung[['시가']]

# 11월 20일 데이터 삭제
samsung_target.drop(samsung_target.index[-1], inplace=True)
hite.drop(hite.index[-1], inplace=True)
samsung_data.drop(samsung_data.index[-1], inplace=True)

#to numpy
samsung_target=samsung_target.to_numpy()
samsung_data=samsung_data.to_numpy()
hite_target=hite.to_numpy()
gold_target=gold.to_numpy()
kosdaq_target=kosdaq.to_numpy()


#데이터 스케일링
from sklearn. preprocessing import StandardScaler, MinMaxScaler
scaler1=StandardScaler()
scaler1.fit(samsung_target)
samsung_target=scaler1.transform(samsung_target)

scaler2=StandardScaler()
scaler2.fit(hite_target)
hite_target=scaler2.transform(hite_target)

scaler3=StandardScaler()
scaler3.fit(gold_target)
gold_target=scaler3.transform(gold_target)

scaler4=MinMaxScaler()
scaler4.fit(kosdaq_target)
kosdaq_target=scaler4.transform(kosdaq_target)

# x 데이터 다섯개씩 자르기
size=5
samsung_target=split_data(samsung_target, size)
hite_target=split_data(hite_target, size)
gold_target=split_data(gold_target, size)
kosdaq_target=split_data(kosdaq_target, size)
hite_target=hite_target[:samsung_target.shape[0],:]
gold_target=gold_target[:samsung_target.shape[0],:]
kosdaq_target=kosdaq_target[:samsung_target.shape[0],:]

# y 데이터 추출
samsung_data=samsung_data[size+1:, :]

# predict 데이터 추출
samsung_target_predict=samsung_target[-1]
hite_target_predict=hite_target[-1]
gold_target_predict=gold_target[-1]
kosdaq_target_predict=kosdaq_target[-1]

samsung_target=samsung_target[:-2, :, :]
hite_target=hite_target[:-2, :, :]
gold_target=gold_target[:-2, :, :]
kosdaq_target=kosdaq_target[:-2, :, :]

print(samsung_target.shape) # (620, 5, 4)
print(hite_target.shape) #(620, 5, 5)
print(gold_target.shape) #(620, 5, 6)
print(kosdaq_target.shape) #(620, 5, 3)
print(samsung_data.shape) # (620, 1)

samsung_target=samsung_target.astype('float32')
samsung_data=samsung_data.astype('float32')
samsung_target_predict=samsung_target_predict.astype('float32')
hite_target=hite_target.astype('float32')
hite_target_predict=hite_target_predict.astype('float32')
gold_target=gold_target.astype('float32')
gold_target_predict=gold_target_predict.astype('float32')

np.save('./data/samsung_target.npy', arr=samsung_target)
np.save('./data/samsung_target_predict.npy', arr=samsung_target_predict)
np.save('./data/samsung_data.npy', arr=samsung_data)
np.save('./data/hite_target.npy', arr=hite_target)
np.save('./data/hite_target_predict.npy', arr=hite_target_predict)
np.save('./data/gold_target.npy', arr=gold_target)
np.save('./data/gold_target_predict.npy', arr=gold_target_predict)
np.save('./data/kosdaq_target.npy', arr=kosdaq_target)
np.save('./data/kosdaq_target_predict.npy', arr=kosdaq_target_predict)

# train, test 분리
from sklearn.model_selection import train_test_split
samsung_target_train, samsung_target_test, samsung_data_train, samsung_data_test=train_test_split(samsung_target, samsung_data, train_size=0.8)
hite_target_train, hite_target_test, gold_target_train, gold_target_test, kosdaq_target_train, kosdaq_target_test=train_test_split(hite_target, gold_target, kosdaq_target, train_size=0.8)

samsung_target_predict=samsung_target_predict.reshape(1,5,4)
hite_target_predict=hite_target_predict.reshape(1,5,5)
gold_target_predict=gold_target_predict.reshape(1,5,6)
kosdaq_target_predict=kosdaq_target_predict.reshape(1,5,3)

######### 2. LSTM 회귀모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
samsung_input1=Input(shape=(5,4))
samsung_1=LSTM(1000, activation='relu')(samsung_input1)
samsung_2=Dense(600, activation='relu')(samsung_1)
samsung_3=Dense(1000, activation='relu')(samsung_2)
samsung_4=Dense(2000, activation='relu')(samsung_3)
samsung_5=Dense(900, activation='relu')(samsung_4)
samsung_6=Dense(500, activation='relu')(samsung_5)
samsung_6=Dense(100, activation='relu')(samsung_6)
samsung_6=Dense(50, activation='relu')(samsung_6)
samsung_output=Dense(1)(samsung_6)

hite_input1=Input(shape=(5,5))
hite_1=LSTM(500, activation='relu')(hite_input1)
hite_2=Dense(800,activation='relu')(hite_1)
hite_3=Dense(1500,activation='relu')(hite_2)
hite_4=Dense(900,activation='relu')(hite_3)
hite_5=Dense(600,activation='relu')(hite_4)
hite_6=Dense(300,activation='relu')(hite_5)
hite_7=Dense(200,activation='relu')(hite_6)
hite_8=Dense(100,activation='relu')(hite_7)
hite_output=Dense(1)(hite_8)

gold_input1=Input(shape=(5,6))
gold_1=LSTM(700, activation='relu')(gold_input1)
gold_2=Dense(1500,activation='relu')(gold_1)
gold_3=Dense(2500,activation='relu')(gold_2)
gold_4=Dense(900,activation='relu')(gold_3)
gold_5=Dense(600,activation='relu')(gold_4)
gold_6=Dense(300,activation='relu')(gold_5)
gold_7=Dense(100,activation='relu')(gold_6)
gold_8=Dense(50,activation='relu')(gold_7)
gold_output=Dense(1)(gold_8)

kosdaq_input1=Input(shape=(5,3))
kosdaq_1=LSTM(300, activation='relu')(kosdaq_input1)
kosdaq_2=Dense(600,activation='relu')(kosdaq_1)
kosdaq_3=Dense(900,activation='relu')(kosdaq_2)
kosdaq_4=Dense(700,activation='relu')(kosdaq_3)
kosdaq_5=Dense(800,activation='relu')(kosdaq_4)
kosdaq_6=Dense(400,activation='relu')(kosdaq_5)
kosdaq_7=Dense(200,activation='relu')(kosdaq_6)
kosdaq_8=Dense(50,activation='relu')(kosdaq_7)
kosdaq_output=Dense(1)(hite_8)

merge1=concatenate([samsung_output, hite_output])

output1=Dense(1000)(merge1)
output2=Dense(700)(output1)
output3=Dense(500)(output2)
output4=Dense(250)(output3)
output5=Dense(80)(output4)
output6=Dense(1)(output5)

model=Model(inputs=[samsung_input1, hite_input1, gold_input1, kosdaq_input1], outputs=output6)

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',  patience=256, mode='auto')
modelpath='./model/samsung-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit([samsung_target_train, hite_target_train, gold_target_train, kosdaq_target_train], samsung_data_train, epochs=10000, batch_size=100, validation_split=0.2, callbacks=[es, cp])


#4. 평가, 예측
loss=model.evaluate([samsung_target_test, hite_target_test, gold_target_test, kosdaq_target_test], samsung_data_test, batch_size=100)
samsung_data_predict=model.predict([samsung_target_predict, hite_target_predict, gold_target_predict, kosdaq_target_predict])

print("loss : ", loss)
print("월삼성시가 :" , samsung_data_predict)