import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

view_size = 3

def split_x2(seq, size):
    bbb = []
    for i in range(len(seq) - size + 1):
        bbb.append(seq[i:(i+size)])
    return np.array(bbb)

# 1. 데이터
# 1.1 load_data
import numpy as np
samsung_data = np.load('./data/samsung_data.npy', allow_pickle=True).astype('float32')
samsung_data = split_x2(samsung_data, view_size)
samsung_data = samsung_data[:-1]

samsung_target = np.load('./data/samsung_target.npy', allow_pickle=True).astype('float32')
samsung_target = samsung_target[view_size:]

hite_data = np.load('./data/hite_data.npy', allow_pickle=True).astype('float32')
hite_data = split_x2(hite_data, view_size)
hite_data = hite_data[:-1]

hite_target = np.load('./data/hite_target.npy', allow_pickle=True).astype('float32')
hite_target = hite_target[view_size:]


print("samsung_data.shape:", samsung_data.shape)
print('samsung_target.shape',samsung_target.shape)
print("hite_data.shape:", hite_data.shape)
print('hite_target.shape',hite_target.shape)

# 1.2 train_test_split
from sklearn.model_selection import train_test_split 
samsung_data_train,samsung_data_test, samsung_target_train,samsung_target_test = train_test_split(
    samsung_data, samsung_target, train_size=0.9, test_size=0.1, random_state = 44)

hite_data_train,hite_data_test, hite_target_train,hite_target_test = train_test_split(
    hite_data, hite_target, train_size=0.9, test_size=0.1, random_state = 44)

def scaling3D(data, scaler):
    temp_data = data
    num_sample   = data.shape[0] 
    num_sequence = data.shape[1]
    num_feature  = data.shape[2]
    for ss in range(num_sequence):
        scaler.fit(data[:, ss, :])

    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(data[:, ss, :]).reshape(num_sample, 1, num_feature))
    temp_data = np.concatenate(results, axis=1)
    return temp_data

def transform3D(data, scaler):
    temp_data = data
    num_sample   = data.shape[0]
    num_sequence = data.shape[1]
    num_feature  = data.shape[2]
    results = []
    for ss in range(num_sequence):
        results.append(scaler.transform(data[:, ss, :]).reshape(num_sample, 1, num_feature))
    temp_data = np.concatenate(results, axis=1)
    return temp_data

# 1.3 scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
samsung_scaler = StandardScaler()
samsung_data_train = scaling3D(samsung_data_train, samsung_scaler)
samsung_data_test = transform3D(samsung_data_test, samsung_scaler)
print("after scaled samsung_data_train.shape:",samsung_data_train.shape)
print("after scaled samsung_data_test.shape:",samsung_data_test.shape)

hite_scaler = StandardScaler()
hite_data_train = scaling3D(hite_data_train, hite_scaler)
hite_data_test = transform3D(hite_data_test, hite_scaler)
print("after scaled hite_data_train.shape",hite_data_train.shape)
print("after scaled hite_data_test.shape:",hite_data_test.shape)

# 1.4 reshape
samsung_data_train = samsung_data_train.reshape(samsung_data_train.shape[0],samsung_data_train.shape[1]*samsung_data_train.shape[2])
samsung_data_test = samsung_data_test.reshape(samsung_data_test.shape[0],samsung_data_test.shape[1]*samsung_data_test.shape[2])
print("after reshape x:", samsung_data_train.shape, samsung_data_test.shape)

hite_data_train = hite_data_train.reshape(hite_data_train.shape[0],hite_data_train.shape[1]*hite_data_train.shape[2])
hite_data_test = hite_data_test.reshape(hite_data_test.shape[0],hite_data_test.shape[1]*hite_data_test.shape[2])
print("after reshape x:", hite_data_train.shape, hite_data_test.shape)

# 2.모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

samsung_model_input = Input(shape=(samsung_data_train.shape[1],))
dense1 = Dense(20, activation='relu', name='dense1_1')(samsung_model_input)
dense1 = Dense(15, activation='relu', name='dense1_2')(dense1)
dense1 = Dense(10, activation='relu', name='dense1_3')(dense1)
samsung_model_output1 = Dense(1, name='samsung_model_output1')(dense1)

hite_model_input = Input(shape=(hite_data_train.shape[1],))
dense2 = Dense(30, activation='relu', name='dense2_1')(hite_model_input)
dense2 = Dense(14, activation='relu', name='dense2_2')(dense2)
dense2 = Dense(10, activation='relu', name='dense2_3')(dense2)
hite_model_output1 = Dense(1, name='hite_model_output1')(dense2)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([samsung_model_output1, hite_model_output1])
middle = Dense(30, name='middle1')(merge1)
middle = Dense(17, name='middle2')(middle)
middle = Dense(11, name='middle3')(middle)

samsung_out = Dense(30, name='output1_1')(middle)
samsung_out = Dense(7, name='output1_2')(samsung_out)
samsung_model_output2 = Dense(1, name='output1_3')(samsung_out)


total_model = Model(inputs=[samsung_model_input,hite_model_input], 
                    outputs=samsung_model_output2)
total_model.summary()

modelpath = './model/hcbae22_{epoch:02d}_{val_loss:.4f}.hdf5'
model_save_path = "./save/hcbae22_model.h5"
weights_save_path = './save/hcbae22_weights.h5'

# 3. 컴파일, 훈련
total_model.compile(loss='mse', optimizer='adam',metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='val_loss',patience=50,mode='auto',verbose=2)

from tensorflow.keras.callbacks import ModelCheckpoint
model_check_point = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

total_model.fit([samsung_data_train, hite_data_train], samsung_target_train,epochs=1000,batch_size=128,validation_split=0.2, verbose=1)
total_model.save(model_save_path)
total_model.save_weights(weights_save_path)

# 4. 평가, 예측
result = total_model.evaluate([samsung_data_test, hite_data_test],samsung_target_test, batch_size=128)
print("loss: ", result[0])
print("mae: ", result[1])

y_predict = total_model.predict([samsung_data_test, hite_data_test])
print("y_predict:", y_predict)

y_recovery = samsung_target_test

# 사용자정의 RMSE 함수
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE:", RMSE(y_recovery, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_recovery, y_predict)
print("R2:", r2)
