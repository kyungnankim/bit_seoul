import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np #데이터 처리
import pandas as pd #데이터 처리
import warnings
warnings.filterwarnings('ignore')
from collections import Counter # count 용도
import matplotlib as sns
import matplotlib.pyplot as plt # 시각화
import seaborn as sns #시각화
import folium # 지도 관련 시각화
from folium.plugins import MarkerCluster #지도 관련 시각화
import geopy.distance #거리 계산해주는 패키지 사용
import random #데이터 샘플링
from sklearn.model_selection import GridSearchCV #모델링
from sklearn.ensemble import RandomForestRegressor #모델링
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import folium # 지도 관련 시각화
from folium.plugins import MarkerCluster #지도 관련 시각화
import geopy.distance #거리 계산해주는 패키지 사용
from geopy import distance
import random #데이터 샘플링

import matplotlib.pyplot as plt # 시각화
import seaborn as sns #시각화

import folium # 지도 관련 시각화
from folium.plugins import MarkerCluster #지도 관련 시각화
import geopy.distance #거리 계산해주는 패키지 사용

from sklearn.model_selection import GridSearchCV #모델링
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
test2 = pd.read_csv('./mini_project/model/test_dist_name.csv', encoding='CP949', header=0,index_col=None,sep=',')
train22= pd.read_csv('./mini_project/model/train_dist_name.csv',encoding='CP949', header=0,index_col=None,sep=',')

train22 = train22.drop(['date','precipitation','bus_route_id','in_out','station_code','station_name','latitude','longitude','6~7_ride','7~8_ride','8~9_ride','9~10_ride','10~11_ride','11~12_ride','6~7_takeoff','7~8_takeoff','8~9_takeoff','9~10_takeoff','10~11_takeoff','11~12_takeoff','location','dist_jeju','dist_gosan','dist_seongsan','dist_po','dust_name','temp'],axis=1)
test2 = test2.drop(['date','bus_route_id','in_out','station_code','station_name','latitude','longitude','6~7_ride','7~8_ride','8~9_ride','9~10_ride','10~11_ride','11~12_ride','6~7_takeoff','7~8_takeoff','8~9_takeoff','9~10_takeoff','10~11_takeoff','11~12_takeoff','location','dist_jeju','dist_gosan','dist_seongsan','dist_po','dust_name','temp'],axis=1)
print(train22)
print(test2)
		
# train22 = train22.groupby(['68a', '810a', '1012a','18~20_ride'])['precipitation'].sum().reset_index()
# print(train22)

train_target=train22[['68a', '810a', '1012a','18~20_ride']]
train_data=train22[['precipitation']]

test_target=test2[['68a', '810a', '1012a']]
test_data=test2[['precipitation']]
# test2_target=test2[['68a', '810a', '1012a']]
# test2_data=test2[['precipitation']]
test_target = test_target.astype(int)
test_data = test_data.astype(int)
train_target=train_target[:10000,1:]
train_data=train_data[:10000,:]
test_target=test_target[:10000,:]
test_data=test_data[:10000,:]
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
X = train_target
y = test_data
sns.distplot( np.log1p(train_target ), color='red' , label='train_target')
sns.distplot( np.log1p( train_data ) , color='yellow', label='train_data')
plt.show()
# train_target = train_target.to_numpy()
# np.save('./mini_project/data/train_target.npy',arr=train_target)
# train_data = train_data.to_numpy()
# np.save('./mini_project/data/train_data.npy',arr=train_data)

# test_target = test_target.to_numpy()
# np.save('./mini_project/data/test_target.npy',arr=test_target)
# test_data = test_data.to_numpy()
# np.save('./mini_project/data/test_data.npy',arr=test_data)




# from sklearn. preprocessing import StandardScaler, MinMaxScaler
# scaler1=StandardScaler()
# scaler1.fit(train22_target)
# train22_target=scaler1.transform(train22_target)

# scaler2=StandardScaler()
# scaler2.fit(test2)
# test2=scaler2.transform(test2)




