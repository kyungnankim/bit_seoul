import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd

# def split_data(x, size) :
#     data=[]
#     for i in range(x.shape[0]-size+1) :
#         data.append(x[i:i+size,:])
#     return np.array(data)

ride = pd.read_csv('./mini_project/2020년_버스노선별_정류장별_시간대별_승하차_인원_정보(01월).csv', header=0, index_col=None, encoding='CP949',sep=',')
route = pd.read_csv('./mini_project/BUS_STATION_BOARDING_MONTH_202001.csv', header=0, index_col=None, encoding='CP949',sep=',' )
humidity = pd.read_csv('./mini_project/humidity/중앙동_습도_202001_202001.csv', header=0, index_col=None, encoding='CP949', sep=',' )

#정렬을 일자별 오름차순으로 변경
# ride=ride.sort_values(['사용년월'], ascending=['True'])
# route=route.sort_values(['사용일자'], ascending=['True'])
# humidity=humidity.sort_values(['day'], ascending=['True'])

#필요한 컬럼만
ride=ride[['사용년월','노선번호','버스정류장ARS번호','6시승차총승객수','7시승차총승객수','8시승차총승객수']]
route=route[['사용일자','버스정류장ARS번호','승차총승객수','하차총승객수']]
humidity=humidity[['day','hour','value location']]
print(humidity)
ride_target=ride[['사용년월','노선번호','버스정류장ARS번호','6시승차총승객수','7시승차총승객수']]
ride_data=ride[['8시승차총승객수']]

#콤마 제거 후 문자를 정수로 변환
# for i in range(len(ride.index)) :
#     for j in range(len(ride.iloc[i])) :
#         ride.iloc[i, j]=int(ride.iloc[i, j].replace('~', ''))

# for i in range(len(route.index)) :
#     for j in range(len(route.iloc[i])) :
#         route.iloc[i, j]=int(route.iloc[i, j].replace('~', ''))

# for i in range(len(humidity.index)) :
#     for j in range(len(humidity.iloc[i])) :
#         humidity.iloc[i, j]=int(humidity.iloc[i, j].replace(',', ''))
# ride_target=ride[['사용년월', '노선번호', '버스정류장ARS번호', '역명', '6시승차총승객수', '7시승차총승객수']]
# ride_data=ride[['8시승차총승객수']]

#to numpy
ride_target=ride_target.to_numpy()
ride_data=ride_data.to_numpy()
route_target=route.to_numpy()
humidity_target=humidity.to_numpy()


# #데이터 스케일링
from sklearn. preprocessing import StandardScaler, MinMaxScaler
scaler1=StandardScaler()
scaler1.fit(ride_target)
ride_target=scaler1.transform(ride_target)

scaler2=StandardScaler()
scaler2.fit(route_target)
route_target=scaler2.transform(route_target)

scaler3=StandardScaler()
scaler3.fit(humidity_target)
humidity_target=scaler3.transform(humidity_target)

# # x 데이터 다섯개씩 자르기
# size=5
# ride_target=split_data(ride_target, size)
# route_target=split_data(route_target, size)
# humidity_target=split_data(humidity_target, size)

# route_target=route_target[:ride_target.shape[0],:]
# humidity_target=humidity_target[:ride_target.shape[0],:]

# # y 데이터 추출
# ride_data=ride_data[size+1:, :]



# # predict 데이터 추출
# ride_target_predict=ride_target[-1]
# route_target_predict=route_target[-1]
# humidity_target_predict=humidity_target[-1]

# ride_target=ride_target[:-2, :, :]
# route_target=route_target[:-2, :, :]
# humidity_target=humidity_target[:-2, :, :]

# print(ride_target.shape) # (620, 5, 4)
# print(route_target.shape) #(620, 5, 5)
# print(humidity_target.shape) #(620, 5, 6)
# print(ride_data.shape) # (620, 1)

# ride_target=ride_target.astype('float32')
# ride_data=ride_data.astype('float32')
# route_target=route_target.astype('float32')
# route_target_predict=route_target_predict.astype('float32')
# humidity_target=humidity_target.astype('float32')
# humidity_target_predict=humidity_target_predict.astype('float32')

# ride_target_predict=ride_target_predict.astype('float32')

# np.save('/data/ride_target.npy', arr=ride_target)
# np.save('/data/ride_target_predict.npy', arr=ride_target_predict)
# np.save('/data/ride_data.npy', arr=ride_data)
# np.save('/data/route_target.npy', arr=route_target)
# np.save('/data/route_target_predict.npy', arr=route_target_predict)
# np.save('/data/humidity_target.npy', arr=humidity_target)
# np.save('/data/humidity_target_predict.npy', arr=humidity_target_predict)


# # train, test 분리
# from sklearn.model_selection import train_test_split
# ride_target_train, ride_target_test, ride_data_train, ride_data_test=train_test_split(ride_target, ride_data, train_size=0.8)
# route_target_train, route_target_test, humidity_target_train, humidity_target_test =train_test_split(route_target, humidity_target, train_size=0.8)

# ride_target_predict=ride_target_predict.reshape(1,5,4)
# route_target_predict=route_target_predict.reshape(1,5,5)
# humidity_target_predict=humidity_target_predict.reshape(1,5,6)

# ######### 2. LSTM 회귀모델
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
# ride_input1=Input(shape=(5,4))
# ride_1=LSTM(1000, activation='relu')(ride_input1)
# ride_2=Dense(600, activation='relu')(ride_1)
# ride_3=Dense(1000, activation='relu')(ride_2)
# ride_4=Dense(2000, activation='relu')(ride_3)
# ride_5=Dense(900, activation='relu')(ride_4)
# ride_6=Dense(500, activation='relu')(ride_5)
# ride_6=Dense(100, activation='relu')(ride_6)
# ride_6=Dense(50, activation='relu')(ride_6)
# ride_output=Dense(1)(ride_6)

# route_input1=Input(shape=(5,5))
# route_1=LSTM(500, activation='relu')(route_input1)
# route_2=Dense(800,activation='relu')(route_1)
# route_3=Dense(1500,activation='relu')(route_2)
# route_4=Dense(900,activation='relu')(route_3)
# route_5=Dense(600,activation='relu')(route_4)
# route_6=Dense(300,activation='relu')(route_5)
# route_7=Dense(200,activation='relu')(route_6)
# route_8=Dense(100,activation='relu')(route_7)
# route_output=Dense(1)(route_8)

# humidity_input1=Input(shape=(5,6))
# humidity_1=LSTM(700, activation='relu')(humidity_input1)
# humidity_2=Dense(1500,activation='relu')(humidity_1)
# humidity_3=Dense(2500,activation='relu')(humidity_2)
# humidity_4=Dense(900,activation='relu')(humidity_3)
# humidity_5=Dense(600,activation='relu')(humidity_4)
# humidity_6=Dense(300,activation='relu')(humidity_5)
# humidity_7=Dense(100,activation='relu')(humidity_6)
# humidity_8=Dense(50,activation='relu')(humidity_7)
# humidity_output=Dense(1)(humidity_8)

# merge1=concatenate([ride_output, route_output])

# output1=Dense(1000)(merge1)
# output2=Dense(700)(output1)
# output3=Dense(500)(output2)
# output4=Dense(250)(output3)
# output5=Dense(80)(output4)
# output6=Dense(1)(output5)

# model=Model(inputs=[ride_input1, route_input1, humidity_input1], outputs=output6)

# model.summary()

# #3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# model.compile(loss='mse', optimizer='adam')
# es=EarlyStopping(monitor='val_loss',  patience=256, mode='auto')
# modelpath='./model/ride-{epoch:02d}-{val_loss:.4f}.hdf5'
# cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# model.fit([ride_target_train, route_target_train, humidity_target_train], ride_data_train, epochs=100000, batch_size=100, validation_split=0.2, callbacks=[es, cp])


# #4. 평가, 예측
# loss=model.evaluate([ride_target_test, route_target_test, humidity_target_test], ride_data_test, batch_size=100)
# ride_data_predict=model.predict([ride_target_predict, route_target_predict, humidity_target_predict])

# print("loss : ", loss)
# print("월삼성시가 :" , ride_data_predict)