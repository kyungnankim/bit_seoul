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

nine = pd.read_csv('./mini_project/data/2020년_버스노선별_정류장별_시간대별_승하차_인원_정보(10월).npy', header=0, index_col=None, encoding='CP949',sep=',')
ten = pd.read_csv('./mini_project/data/BUS_STATION_BOARDING_MONTH_202001.csv.npy', header=0, index_col=None, encoding='CP949',sep=',' )
humidity = pd.read_csv('./mini_project/data/humidity/중앙동_습도_202001_202001.csv.npy', header=0, index_col=None, encoding='CP949', sep=',' )

route=route[['사용일자','버스정류장ARS번호','승차총승객수','하차총승객수']]
humidity=humidity[['day','hour','value location']]

ride=ride[['사용년월','노선번호','버스정류장ARS번호','6시승차총승객수','7시승차총승객수','8시승차총승객수']]
ride_target=ride[['사용년월','노선번호','버스정류장ARS번호','6시승차총승객수','7시승차총승객수']]
ride_data=ride[['8시승차총승객수']]

y = ride['8시승차총승객수']
x = ride.drop('8시승차총승객수',axis=1)

print(x.shape)#(4898, 11)
print(y.shape)#(4898, )

# 모델 만든거 이어라
print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66,shuffle=True, train_size=0.8)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)

#####2. 모델
model = RandomForestRegressor()

####3. 훈련
model.fit(x_train, y_train)

#####4. 평가, 예측
score = model.score(x_test, y_test)
print("model.score : ", score)

#accuracy_score를 넣어서 비교할 것
#회귀모델일 경우  r2_score로 코딩해서 score와 비교할 것
y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2=r2_score(y_test, y_predict)

print("R2 : ", r2)
