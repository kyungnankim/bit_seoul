import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

one = pd.read_csv('./mini_project/2020년_버스노선별_정류장별_시간대별_승하차_인원_정보(01~09월).csv', header=0, index_col=None, encoding='CP949',sep=',')
ten = pd.read_csv('./mini_project/2020년_버스노선별_정류장별_시간대별_승하차_인원_정보(10월).csv', header=0, index_col=None, encoding='CP949',sep=',' )
humidity = pd.read_csv('./mini_project/humidity/중앙동_습도_202001_202001.csv', header=0, index_col=None, encoding='CP949', sep=',' )

print(type(one))
print(type(ten))
print(type(humidity))
print(one.head())
print(one.head(10))
print(one.shape)
print(one.isnull().sum())
print(ten.isnull().sum())
print(humidity.isnull().sum())
# one = one.drop(['노선명', '표준버스정류장ID'],axis=1)
# ten = ten.drop(['노선명', '표준버스정류장ID'],axis=1)

# st_name_mapping = {"봉천사거리": 0, "신논현역.구교보타워사거리": 1}
# one['st_name'] = one['st_name'].map(st_name_mapping)
# ten['st_name'] = ten['st_name'].map(st_name_mapping)

one['date'] = pd.to_datetime(one['date'])
one = pd.get_dummies(one,columns=['date'])

ten['date'] = pd.to_datetime(ten['date'])
ten = pd.get_dummies(ten,columns=['date'])
one['st_name'].value_counts()


one['st_name'] = one['st_name'].map({'봉천사거리':0,'신논현역.구교보타워사거리':1})
ten['st_name'] = ten['st_name'].map({'봉천사거리':0,'신논현역.구교보타워사거리':1})


coords_jejusi = (33.500770, 126.522761) #제주시의 위도 경도
coords_seoquipo = (33.259429, 126.558217) #서귀포시의 위도 경도


train['dis_jejusi'] = [geopy.distance.vincenty((train['latitude'].iloc[i],train['longitude'].iloc[i]), coords_jejusi).km for i in range(len(train))]
train['dis_seoquipo'] = [geopy.distance.vincenty((train['latitude'].iloc[i],train['longitude'].iloc[i]), coords_seoquipo).km for i in range(len(train))]

train['dis_jejusi'].corr(train['18~20_ride'])

























# print(one.shape())#[5 rows x 24 columns]
# one=one.sort_values(['사용년월'], ascending=['True'])
# ten=ten.sort_values(['사용년월'], ascending=['True'])
# humidity=humidity.sort_values(['day'], ascending=['True'])

# one = one[['사용년월','노선번호','버스정류장ARS번호','역명','6시승차총승객수','7시승차총승객수','8시승차총승객수','9시승차총승객수']]
# ten = ten[['사용년월','노선번호','버스정류장ARS번호','역명','6시승차총승객수','7시승차총승객수','8시승차총승객수','9시승차총승객수']]
# humidity = humidity[['day','hour','value location']]
# # print("ten.head()",ten.shape())#[5 rows x 24 columns]

# # ten=ten[['사용일자','버스정류장ARS번호','승차총승객수','하차총승객수']]
# # one['date'] = pd.to_datetime(one['date'])
# one= one[['사용년월','노선번호','버스정류장ARS번호','역명','6시승차총승객수','7시승차총승객수','8시승차총승객수','9시승차총승객수']]
# one_target= one[['사용년월','버스정류장ARS번호','역명','6시승차총승객수','7시승차총승객수','8시승차총승객수','9시승차총승객수']]
# one_data=one[['노선번호']]
# # for i in range(len(one.index)) :
# #     for j in range(len(one.iloc[i])) :
# #         one.iloc[i, j]=int(one.iloc[i, j].replace('&', ''))

# # for i in range(len(ten.index)) :
# #     for j in range(len(ten.iloc[i])) :
# #         ten.iloc[i, j]=int(ten.iloc[i, j].replace('&', ''))
# one_target= one[['사용년월','버스정류장ARS번호','역명','6시승차총승객수','7시승차총승객수','8시승차총승객수','9시승차총승객수']]
# one_data=one[['노선번호']]
# # print(one_target.shape())
# print(type(one))
# print(type(ten))
# print(type(one))
# #to numpy

# one=one.to_numpy()
# one_target=one_target.to_numpy()
# one_data=one_data.to_numpy()
# ten_target=ten.to_numpy()
# humidity_target=humidity.to_numpy()

# print(type(one.data))
# print(one.shape)
# print(one_target.shape)
# print(one_data.shape)
# print(ten.shape)
# print(humidity.shape)
# # from sklearn. preprocessing import StandardScaler, MinMaxScaler
# # scaler1=StandardScaler()
# # scaler1.fit(one_target)
# # one_target=scaler1.transform(one_target)
# # one_target = one_target.reshape(one_target.shape[1:])
# # one_data = one_data.reshape(one_data.shape[1:])
# # ten_target = ten_target.reshape(ten_target.shape[1:])
# # one_target = one_target.transpose()
# # one_data = one_data.transpose()
# # ten_target = ten_target.transpose()
# np.save('./mini_project/data/one_target.npy', arr=one_target)
# np.save('./mini_project/data/one_data.npy', arr=one_data)
# np.save('./mini_project/data/ten_target.npy', arr=ten_target)
# np.save('./mini_project/data/humidity_target.npy', arr=humidity_target)
# # print(one_target.shape)
# # print(one_data.shape)
# # print(ten_target.shape)
# # print(humidity_target.shape)
# # # scaler2=StandardScaler()
# # # scaler2.fit(ten_target)
# # # ten_target=scaler2.transform(ten_target)

# # # scaler3=StandardScaler()
# # # scaler3.fit(humidity_target)
# # # humidity_target=scaler3.transform(humidity_target)

# # # samsung_target=samsung_target.astype('float32')
# # # samsung_data=samsung_data.astype('float32')
# # # samsung_target_predict=samsung_target_predict.astype('float32')
# # # hite_target=hite_target.astype('float32')
# # # hite_target_predict=hite_target_predict.astype('float32')
# # # gold_target=gold_target.astype('float32')
# # # gold_target_predict=gold_target_predict.astype('float32')

# # # # one_target_train, one_target_test, one_data_train, one_data_test=train_test_split(one_target, one_data, train_size=0.8)
# # # # ten_target_train, ten_target_test, humidity_target_train, humidity_target_test=train_test_split(ten_target, humidity_target,  train_size=0.8)
# # # size=5
# # # one_target=split_data(one_target, size)
# # # ten_target=split_data(ten_target, size)
# # # humidity_target=split_data(humidity_target, size)

# # # ten_target=ten_target[:one_target.shape[0],:]
# # # humidity_target=humidity_target[:one_target.shape[0],:]
# # # print(x_train)
# # # scale=StandardScaler()
# # # scale.fit(x_train)
# # # x_train=scale.transform(x_train)
# # # x_test=scale.transform(x_test)

# # # # #####2. 모델
# # # model = RandomForestRegressor()

# # # # ####3. 훈련
# # # model.fit(x_train, y_train)

# # # #####4. 평가, 예측
# # # score = model.score(x_test, y_test)
# # # print("model.score : ", score)

# # # #accuracy_score를 넣어서 비교할 것
# # # #회귀모델일 경우  r2_score로 코딩해서 score와 비교할 것
# # # y_predict = model.predict(x_test)
# # # from sklearn.metrics import mean_squared_error 
# # # def RMSE(y_test, y_pred) :
# # #     return np.sqrt(mean_squared_error(y_test, y_pred))

# # # print("RMSE : ", RMSE(y_test, y_predict))

# # # from sklearn.metrics import r2_score 
# # # r2=r2_score(y_test, y_predict)

# # # print("R2 : ", r2)
