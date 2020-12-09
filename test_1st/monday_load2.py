import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from tensorflow.keras.models import load_model

# 1. 데이터
samsung_target=np.load('./data/samsung_target.npy')
samsung_target_predict=np.load('./data/samsung_target_predict.npy')
samsung_data=np.load('./data/samsung_data.npy')
hite_target=np.load('./data/hite_target.npy')
hite_target_predict=np.load('./data/hite_target_predict.npy')
gold_target=np.load('./data/gold_target.npy')
gold_target_predict=np.load('./data/gold_target_predict.npy')
kosdaq_target=np.load('./data/kosdaq_target.npy')
kosdaq_target_predict=np.load('./data/kosdaq_target_predict.npy')

# train, test 분리
from sklearn.model_selection import train_test_split
samsung_target_train, samsung_target_test, samsung_data_train, samsung_data_test=train_test_split(samsung_target, samsung_data, train_size=0.8)
hite_target_train, hite_target_test, gold_target_train, gold_target_test, kosdaq_target_train, kosdaq_target_test=train_test_split(hite_target, gold_target, kosdaq_target, train_size=0.8)

samsung_target_predict=samsung_target_predict.reshape(1,5,4)
hite_target_predict=hite_target_predict.reshape(1,5,5)
gold_target_predict=gold_target_predict.reshape(1,5,6)
kosdaq_target_predict=kosdaq_target_predict.reshape(1,5,3)


model = load_model('./model/samsung-1468-32576999424.0000.hdf5')
model_save_path = "./save/samsungmon_model.h5"
weights_save_path = './save/samsungmon_weights.h5'
#4. 평가, 예측
loss=model.evaluate([samsung_target_test, hite_target_test, gold_target_test, kosdaq_target_test], samsung_data_test, batch_size=100)
samsung_data_predict=model.predict([samsung_target_predict, hite_target_predict, gold_target_predict, kosdaq_target_predict])

print("loss : ", loss)
print("monday삼성시가 :" , samsung_data_predict)


