import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#model
import lightgbm as lgbm 
import matplotlib.pyplot as plt # 시각화
import numpy as np #데이터 처리
import pandas as pd #데이터 처리
import warnings
warnings.filterwarnings('ignore')
from collections import Counter # count 용도
import matplotlib as sns
import geopy.distance #거리 계산해주는 패키지 사용
from geopy import distance
import random #데이터 샘플링
from sklearn.model_selection import GridSearchCV #모델링
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
# 지도 관련 시각화 
import folium 
from folium.plugins import MarkerCluster 

#데이터 샘플링
import random
from sklearn.preprocessing import LabelEncoder #인코딩
from sklearn.preprocessing import OneHotEncoder

#validation
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import folium # 지도 관련 시각화
from folium.plugins import MarkerCluster #지도 관련 시각화

#train 18~20_ride encoding='CP949',
####데이터
test = pd.read_csv('./mini_project/data/test.csv', header=0, index_col=None,sep=',')
train = pd.read_csv('./mini_project/data/train.csv', header=0, index_col=None,sep=',' )
# test = test.drop(['station_name','id'],axis=1)
# train = train.drop(['station_name','id'],axis=1)
print(train['date'].agg(['min','max'])) #2019-09-01 ~ 2019-09-30 (train)
print(test['date'].agg(['min','max']))
# train['date2'] = pd.to_datetime(train['date'])
# train = pd.get_dummies(train,columns=['weekday'])

# test['date2'] = pd.to_datetime(test['date'])
# test['weekday'] = test['date2'].dt.weekday
# test = pd.get_dummies(test,columns=['weekday'])

# del train['date2']
# del test['date2']

train['in_out'].value_counts()

train['in_out'] = train['in_out'].map({'시내':0,'시외':1})
test['in_out'] = test['in_out'].map({'시내':0,'시외':1})

train['68a']=train['6~7_ride']+train['7~8_ride'] # 6 ~ 8시 승차인원
train['810a']=train['8~9_ride']+train['9~10_ride']
train['1012a']=train['10~11_ride']+train['11~12_ride']


train22=train[['68a','810a','1012a','18~20_ride']]

test['68a']=test['6~7_ride']+test['7~8_ride']
test['810a']=test['8~9_ride']+test['9~10_ride']
test['1012a']=test['10~11_ride']+test['11~12_ride']

target_col = train['18~20_ride']
multi_station = train.groupby('station_name')['station_code'].nunique().sort_values()
multi_station[multi_station >= 7]
train[train['station_name'].isin(multi_station.index)][['station_code', 'station_name', 'latitude', 'longitude']]

len(train['station_code'].unique()), len(train['station_name'].unique())



import folium # 지도 관련 시각화
####위도경도 가즈아
jeju=(33.51411, 126.52969) # 제주 측정소 근처
gosan=(33.29382, 126.16283) #고산 측정소 근처
seongsan=(33.38677, 126.8802) #성산 측정소 근처
po=(33.24616, 126.5653) #서귀포 측정소 근처

# map_osm= folium.Map((33.399835, 126.506031),zoom_start=9)
# mc = MarkerCluster()

# mc.add_child( folium.Marker(location=jeju,popup='제주 측정소',icon=folium.Icon(color='red',icon='info-sign') ) ) #제주 측정소 마커 생성
# map_osm.add_child(mc) #마커를 map_osm에 추가

# mc.add_child( folium.Marker(location=gosan,popup='고산 측정소',icon=folium.Icon(color='red',icon='info-sign') ) )
# map_osm.add_child(mc) 

# mc.add_child( folium.Marker(location=seongsan,popup='성산 측정소',icon=folium.Icon(color='red',icon='info-sign') ) )
# map_osm.add_child(mc) 

# mc.add_child( folium.Marker(location=po,popup='서귀포 측정소',icon=folium.Icon(color='red',icon='info-sign') ) )
# map_osm.add_child(mc)

# map_osm.save('./mini_project/data/map_osm.html')



#정류장의 위치만 확인하기 위해 groupby를 실행함
# data=train[['latitude','longitude','station_code']].drop_duplicates(keep='first')
# data2=data.groupby(['station_code'])['latitude','longitude'].mean()

# data2.to_csv("./mini_project/data/folium.csv")
# data2=pd.read_csv("./mini_project/data/folium.csv")

# #정류장의 대략적인 위치를 확인하기 위하여, folium map에 해당 정류장을 표시
# for row in data2.itertuples():
#     mc.add_child(folium.Marker(location=[row.latitude, row.longitude], popup=row.station_code)) #마커 생성
#     map_osm.add_child(mc) #마커를 map_osm에 추가
# map_osm

#map_osm.save('./mini_project/data/map_osm_marker.html')

import matplotlib.pyplot as plt # 시각화
import seaborn as sns #시각화


import geopy.distance #거리 계산해주는 패키지 사용
# latitude / longitude를 묶어줘서 location컬럼 생성.
train['location'] = train.apply(lambda row: (row.latitude, row.longitude), axis=1)
train['dist_jeju'] = train['location'].apply(lambda x: geopy.distance.geodesic(x,jeju).km)
train['dist_gosan'] = train['location'].apply(lambda x: geopy.distance.geodesic(x,gosan).km)
train['dist_seongsan'] = train['location'].apply(lambda x: geopy.distance.geodesic(x,seongsan).km)
train['dist_po'] = train['location'].apply(lambda x: geopy.distance.geodesic(x,po).km)

train['dust_name'] = train[['dist_jeju', 'dist_gosan', 'dist_seongsan', 'dist_po']].apply(lambda row: ['dist_jeju', 'dist_gosan', 'dist_seongsan', 'dist_po'][row.argmin()][5:], axis=1)
print(train)
test['location'] = test.apply(lambda row: (row.latitude, row.longitude), axis=1)


#정류장의 위치만 확인하기 위해 groupby 실행
test['location'] = test.apply(lambda row: (row.latitude, row.longitude), axis=1)

test['dist_jeju'] = test['location'].apply(lambda x: geopy.distance.geodesic(x,jeju).km)
test['dist_gosan'] = test['location'].apply(lambda x: geopy.distance.geodesic(x,gosan).km)
test['dist_seongsan'] = test['location'].apply(lambda x: geopy.distance.geodesic(x,seongsan).km)
test['dist_po'] = test['location'].apply(lambda x: geopy.distance.geodesic(x,po).km)

test['dust_name'] = test[['dist_jeju', 'dist_gosan', 'dist_seongsan', 'dist_po']].apply(lambda row: ['dist_jeju', 'dist_gosan', 'dist_seongsan', 'dist_po'][row.argmin()][5:], axis=1)
print(test)

##########################################날씨
weather=pd.read_csv("./mini_project/data/DAY_2019_2019.csv", header=0,encoding='CP949', index_col=None,sep=',')

weather = weather.drop(['강수 계속시간(hr)','최저기온(°C)','최고기온 시각(hhmi)','최고기온(°C)','10분 최다 강수량(mm)','1시간 최다강수량(mm)','최대 순간 풍속(m/s)','최대 순간풍속 시각(hhmi)','최대 풍속 풍향(16방위)',
            '최대 풍속 시각(hhmi)','평균 풍속(m/s)','풍정합(100m)','평균 이슬점온도(°C)','최소 상대습도(%)','최소 상대습도 시각(hhmi)','평균 상대습도(%)','평균 증기압(hPa)','평균 현지기압(hPa)','최고 해면기압(hPa)',
            '최고 해면기압 시각(hhmi)','최저 해면기압(hPa)','최저 해면기압 시각(hhmi)','평균 해면기압(hPa)','가조시간(hr)','합계 일조시간(hr)','1시간 최다일사량(MJ/m2)','합계 일사량(MJ/m2)',
            '일 최심신적설(cm)','일 최심신적설 시각(hhmi)','일 최심적설(cm)','일 최심적설 시각(hhmi)','합계 3시간 신적설(cm)','평균 전운량(1/10)',
            '평균 중하층운량(1/10)','평균 지면온도(°C)','최저 초상온도(°C)','평균 5cm 지중온도(°C)','평균 10cm 지중온도(°C)','평균 20cm 지중온도(°C)','평균 30cm 지중온도(°C)',
            '0.5m 지중온도(°C)','1.0m 지중온도(°C)','1.5m 지중온도(°C)','3.0m 지중온도(°C)','5.0m 지중온도(°C)','합계 대형증발량(mm)','합계 소형증발량(mm)','9-9강수(mm)','안개 계속시간(hr)',
            '최대 풍속(m/s)','1시간 최다일사 시각(hhmi)'],axis=1)


weather['지점'] = [ str(i) for i in weather['지점'] ]
weather['지점'] = ['jeju' if i=='184' else i for i in weather['지점'] ]  # 위도 : 33.51411 경도 : 126.52969
weather['지점'] = ['gosan' if i=='185' else i for i in weather['지점'] ]  # 위도 : 33.29382 경도 : 126.16283
weather['지점'] = ['seongsan' if i=='188' else i for i in weather['지점'] ]  # 위도 : 33.38677 경도 : 126.8802
weather['지점'] = ['po' if i=='189' else i for i in weather['지점'] ]  # 위도 : 33.24616 경도 : 126.5653
rain2 = weather[(weather['일강수량(mm)']>=0) ]
#train/test기간

weather = weather.rename(columns={"일시":"date","지점":"dist_name","일강수량(mm)":"precipitation","평균기온(°C)":"temp"})

# 강수량, 기온를 피쳐로 사용
weather = weather.groupby(['date'])[['temp','precipitation']].sum().reset_index()

#################################################merge#################################
#데이터 합치기
train = pd.merge(train, weather,on=['date'], how='left')
test = pd.merge(test, weather, on=['date'], how='left')

# train['68a']=train['6~7_ride']+train['7~8_ride'] # 6 ~ 8시 승차인원 (2시간단위로 묶어줌)
# train['810a']=train['8~9_ride']+train['9~10_ride']
# train['1012a']=train['10~11_ride']+train['11~12_ride']

# test['68a']=test['6~7_ride']+test['7~8_ride']
# test['810a']=test['8~9_ride']+test['9~10_ride']
# test['1012a']=test['10~11_ride']+test['11~12_ride']

# train2=train[['68a','810a','1012a','18~20_ride']]
# print(train2)
# print(test)
# # cor=train2.corr()
# train2 = pd.get_dummies(train2,columns=['date'])
# test = pd.get_dummies(test,columns=['date'])
# print('train2',train2.shape)#train2 (1061878, 71)
# print('test2 :',test.shape)#test2 : (310121, 70)


# #3가지 시간대로 나눔 : 새벽, 아침, 정오
# dawn_ride_cols = train[['6~7_ride','7~8_ride']]
# morning_ride_cols = train[['8~9_ride','9~10_ride']]
# noon_ride_cols = train[['10~11_ride','11~12_ride']]

# dawn_takoff_cols = test[['6~7_ride','7~8_ride']]
# morning_takeoff_cols = test[['8~9_ride','9~10_ride']]
# noon_takeoff_cols = test[['10~11_ride','11~12_ride']]


# train['avg_dawn_ride'] = train.groupby(['date','bus_route_id']).transform('mean') 
# train['avg_morning_ride'] = train.groupby(['date','bus_route_id']).transform('mean') 
# train['avg_noon_ride'] = train.groupby(['date','bus_route_id']).transform('mean') 

# test['avg_dawn_ride'] = test.groupby(['date','bus_route_id']).transform('mean') 
# test['avg_morning_ride'] = test.groupby(['date','bus_route_id']).transform('mean') 
# test['avg_noon_ride'] = test.groupby(['date','bus_route_id']).transform('mean')  

# train = pd.get_dummies(train,columns=['dist_name'])
# test = pd.get_dummies(test,columns=['dist_name'])


test.to_csv("./mini_project/model/test_dist_namee.csv".index=None)
train2.to_csv("./mini_project/model/train_dist_namee.csv",index=None)

