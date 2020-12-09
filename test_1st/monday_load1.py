import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
view_size = 10

def split_x2(seq, size):
    bbb = []
    for i in range(len(seq) - size + 1):
        bbb.append(seq[i:(i+size)])
    return np.array(bbb)

# 1. 데이터
# load_data
import numpy as np
samsung_data = np.load('./data/samsung_data.npy', allow_pickle=True).astype('float32')
samsung_data = split_x2(samsung_data, view_size)
samsung_pred = samsung_data[-3]
samsung_data = samsung_data[:-2]

samsung_target = np.load('./data/samsung_target.npy', allow_pickle=True).astype('float32')
samsung_target = samsung_target[26:]

hite_data = np.load('./data/hite_data.npy', allow_pickle=True).astype('float32')
hite_data = split_x2(hite_data, view_size)
hite_pred = hite_data[-1]
hite_data = hite_data[:-1]

hite_target = np.load('./data/hite_target.npy', allow_pickle=True).astype('float32')
hite_target = hite_target[view_size:]

gold_data = np.load('./data/gold_data.npy', allow_pickle=True).astype('float32')
gold_data = split_x2(gold_data, view_size)
gold_pred = gold_data[-1]
gold_data = gold_data[:-1]

gold_target = np.load('./data/gold_target.npy', allow_pickle=True).astype('float32')
gold_target = gold_target[view_size:]

kosdaq_data = np.load('./data/kosdaq_data.npy', allow_pickle=True).astype('float32')
kosdaq_data = split_x2(kosdaq_data, view_size)
kosdaq_pred = kosdaq_data[-1]
kosdaq_data = kosdaq_data[:-1]

kosdaq_target = np.load('./data/kosdaq_target.npy', allow_pickle=True).astype('float32')
kosdaq_target = kosdaq_target[view_size:]


print("samsung_data.shape:", samsung_data.shape)
print('samsung_target.shape',samsung_target.shape)
print("hite_data.shape:", hite_data.shape)
print('hite_target.shape',hite_target.shape)
print("gold_data.shape:", gold_data.shape)
print('gold_target.shape',gold_target.shape)
print("kosdaq_data.shape:", kosdaq_data.shape)
print('kosdaq_target.shape',kosdaq_target.shape)

samsung_target_predict=samsung_target_predict.reshape(1,-1)
hite__target_predict=hite__target_predict.reshape(1, -1)
gold__target_predict=gold__target_predict.reshape(1, -1)
kosdak__target_predict=kosdak__target_predict.reshape(1, -1)


# scaler
from sklearn.preprocessing import StandardScaler
samsung_scaler = StandardScaler()
scaler1=StandardScaler()
scaler1.fit(samsung_target)
samsung_target=scaler1.transform(samsung_target)

scaler2=StandardScaler()
scaler2.fit(hite_target)
hite_target=scaler2.transform(hite_target)

scaler3=StandardScaler()
scaler3.fit(gold_target)
gold_target=scaler3.transform(gold_target)

scaler4=StandardScaler()
scaler4.fit(kosdaq_target)
kosdaq_target=scaler4.transform(kosdaq_target)

# x 데이터 다섯개씩 자르기
size=5
def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)
samsung_target=split_data(samsung_target, size)
hite__target=split_data(hite__target, size)
gold__target=split_data(gold__target, size)
kosdak__target=split_data(kosdak__target, size)
hite__target=hite__target[:samsung_target.shape[0],:]
gold__target=gold__target[:samsung_target.shape[0],:]
kosdak__target=kosdak__target[:samsung__target.shape[0],:]

# y 데이터 추출
samsung_data=samsung_data[size+1:, :]



# predict 데이터 추출
samsung__target_predict=samsung__target[-1]
hite__target_predict=hite__target[-1]
gold__target_predict=gold__target[-1]
kosdak__target_predict=kosdak__target[-1]

samsung__target=samsung__target[:-2, :, :]
hite__target=hite__target[:-2, :, :]
gold__target=gold__target[:-2, :, :]
kosdak__target=kosdak__target[:-2, :, :]

print(samsung__target.shape) # (620, 5, 4)
print(hite__target.shape) #(620, 5, 5)
print(gold__target.shape) #(620, 5, 6)
print(kosdak__target.shape) #(620, 5, 3)
print(samsung_data.shape) # (620, 1)

samsung__target=samsung__target.astype('float32')
samsung_data=samsung_data.astype('float32')
samsung__target_predict=samsung__target_predict.astype('float32')
hite__target=hite__target.astype('float32')
hite__target_predict=hite__target_predict.astype('float32')
gold__target=gold__target.astype('float32')
gold__target_predict=gold__target_predict.astype('float32')

np.save('./data/monday/samsung__target.npy', arr=samsung_target)
np.save('./data/monday/samsung_target_predict.npy', arr=samsung_target_predict)
np.save('./data/monday/samsung_data.npy', arr=samsung_data)
np.save('./data/monday/hite__target.npy', arr=hite__target)
np.save('./data/monday/hite__target_predict.npy', arr=hite__target_predict)
np.save('./data/monday/gold__target.npy', arr=gold__target)
np.save('./data/monday/gold__target_predict.npy', arr=gold__target_predict)
np.save('./data/monday/kosdak__target.npy', arr=kosdak__target)
np.save('./data/monday/kosdak__target_predict.npy', arr=kosdak__target_predict)

# train, test 분리
samsung_target_train, samsung_target_test, samsung_data_train, samsung_data_test=train_test_split(samsung_target, samsung_data, train_size=0.8)
hite__target_train, hite__target_test, gold__target_train, gold__target_test, kosdak__target_train, kosdak__target_test=train_test_split(hite__target, gold__target, kosdak__target, train_size=0.8)


# 2.모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM


samsung_model_input = Input(shape=(samsung_data_train.shape[1],1))
dense1 = LSTM(300, activation='relu', name='dense1_1')(samsung_model_input)
dense1 = Dense(400, activation='relu', name='dense1_2')(dense1)
dense1 = Dense(200, activation='relu', name='dense1_3')(dense1)
dense1 = Dense(100, activation='relu', name='dense1_4')(dense1)
samsung_model_output1 = Dense(10, name='samsung_model_output1')(dense1)

hite_model_input = Input(shape=(hite_data_train.shape[1],1))
dense2 = LSTM(300, activation='relu', name='dense2_1')(hite_model_input)
dense2 = Dense(400, activation='relu', name='dense2_2')(dense2)
dense2 = Dense(200, activation='relu', name='dense2_3')(dense2)
dense2 = Dense(100, activation='relu', name='dense2_4')(dense2)
hite_model_output1 = Dense(10, name='hite_model_output1')(dense2)

gold_model_input = Input(shape=(gold_data_train.shape[1],1))
dense3 = LSTM(300, activation='relu', name='dense3_1')(gold_model_input)
dense3 = Dense(400, activation='relu', name='dense3_2')(dense3)
dense3 = Dense(200, activation='relu', name='dense3_3')(dense3)
dense3 = Dense(100, activation='relu', name='dense3_4')(dense3)
gold_model_output1 = Dense(10, name='gold_model_output1')(dense3)

kosdaq_model_input = Input(shape=(kosdaq_data_train.shape[1],1))
dense4 = LSTM(300, activation='relu', name='dense4_1')(kosdaq_model_input)
dense4 = Dense(400, activation='relu', name='dense4_2')(dense4)
dense4 = Dense(200, activation='relu', name='dense4_3')(dense4)
dense4 = Dense(100, activation='relu', name='dense4_4')(dense4)
kosdaq_model_output1 = Dense(10, name='kosdaq_model_output1')(dense4)


from tensorflow.keras.layers import concatenate
merge1 = concatenate([samsung_model_output1, hite_model_output1,
                       gold_model_output1, kosdaq_model_output1])
middle = Dense(400, name='middle1')(merge1)
middle = Dense(200, name='middle2')(middle)
middle = Dense(100, name='middle3')(middle)

samsung_out = Dense(64, name='output1_1')(middle)
samsung_out = Dense(16, name='output1_2')(samsung_out)
samsung_model_output2 = Dense(1, name='output1_3')(samsung_out)

total_model = Model(inputs=[samsung_model_input,hite_model_input,gold_model_input,kosdaq_model_input], 
                    outputs=samsung_model_output2)
total_model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
es=EarlyStopping(monitor='val_loss',  patience=30, mode='auto')
from tensorflow.keras.callbacks import ModelCheckpoint # 모델 체크 포인트
modelpath='./model/samsung-monday-{epoch:02d}-{val_loss:.4f}.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit([samsung__target_train, hite_x_train, gold_x_train, kosdaq_x_train], samsung_data_train, epochs=10000, batch_size=100, validation_split=0.2, callbacks=[es, cp])


#4. 평가, 예측
loss=model.evaluate([samsung__target_test, bit_x_test, gold_x_test, kosdaq_x_test], samsung_data_test, batch_size=100)
samsung_data_predict=model.predict([samsung__target_predict, bit_x_predict, gold_x_predict, kosdaq_x_predict])

print("loss : ", loss)
print("2020.11.23. 월요일 삼성전자 시가 :" , samsung_data_predict)

# predict
samsung_pred = samsung_pred.reshape(1, samsung_pred.shape[0],samsung_pred.shape[1])
samsung_pred = samsung_pred.reshape(1, samsung_pred.shape[1]*samsung_pred.shape[2],1)
print("samsung_pred.shape:",samsung_pred.shape)

hite_pred = hite_pred.reshape(1, hite_pred.shape[0],hite_pred.shape[1])
hite_pred = hite_pred.reshape(1, hite_pred.shape[1]*hite_pred.shape[2],1)
print("hite_pred.shape:",hite_pred.shape)

gold_pred = gold_pred.reshape(1, gold_pred.shape[0],gold_pred.shape[1])
gold_pred = gold_pred.reshape(1, gold_pred.shape[1]*gold_pred.shape[2],1)
print("gold_pred.shape:",gold_pred.shape)

kosdaq_pred = kosdaq_pred.reshape(1, kosdaq_pred.shape[0],kosdaq_pred.shape[1])
kosdaq_pred = kosdaq_pred.reshape(1, kosdaq_pred.shape[1]*kosdaq_pred.shape[2],1)
print("kosdaq_pred.shape:",kosdaq_pred.shape)

samsung_predict_today = total_model.predict([samsung_pred, hite_pred,gold_pred, kosdaq_pred])
print("samsung_predict:", int(samsung_predict_today))

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 단위는 찾아보자
plt.subplot(2,1,1) # 2장 중에 첫 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2장 중에 두 번째
plt.plot(hist.history['mae'], marker='.', c='red')
plt.plot(hist.history['val_mae'], marker='.', c='blue')
plt.grid()
plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epochs')
plt.legend(['mae', 'val_mae'])

plt.show()
