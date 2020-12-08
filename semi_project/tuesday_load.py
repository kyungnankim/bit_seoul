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

train, test의 변수명과 통일시키고, NaN의 값은 0.0000으로 변경
rain3 = rain3.rename(columns={'일강수량(mm)':'avg'}, inplace = True)
rain3 = rain3.rename(columns={'지점':'dist_name'}, inplace = True)
rain3= rain3.fillna(0.00000)
# ##=======================================================================================================================
train2=pd.merge(train,rain3)
test2=pd.merge(test,rain3)
print(rain3)
train2 = pd.get_dummies(train2,columns=['지점'])
test2 = pd.get_dummies(test2,columns=['지점'])


input_var=['in_out','latitude', 'longitude', '68a', '810a', '1012a', '68b', '810b', '1012b',
           'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
           'dis_jeju', 'dis_gosan','dis_seongsan', 'dis_po','평균기온(°C)', '일강수량(mm)', 
           '지점_gosan', '지점_jeju','지점_po', '지점_seongsan']

target=['18~20_ride']

X_train=train2[input_var]
random.seed(1217) #동일한 샘플링하기 위한 시드번호
train_list=random.sample(list(range(X_train.shape[0])), int(round(X_train.shape[0]*0.01,0)) )

X_train=train2[input_var]
X_train=X_train.iloc[train_list,:]
y_train=train2[target]
y_train=y_train.iloc[train_list,:]

X_test=test2[input_var]

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_features': [2,3,5],
    'min_samples_leaf': [2,3],
    'min_samples_split': [2,4,6],
    'n_estimators': [100, 200,500]
}
# Create a based model
rf = RandomForestRegressor(random_state=1217) # 랜덤포레스트 모델을 정의한다.
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid) # GridSearchCV를 정의한다.

grid_search.fit(X_train, y_train)

grid_search.best_params_ #학습 이후 최적의 paramter를 출력

#해당 코드 실행시간 2분 ~ 3분 소요
parameters= [
{'max_features': 5,
 'min_samples_leaf': 2,
 'min_samples_split': 2,
 'n_estimators': 500,
'n_jobs' :-1}
]
#전체 데이터로 적용
X_train=train2[input_var]
y_train=train2[target]

X_test=test2[input_var]

kfold=KFold(n_splits=5, shuffle=True) 
# best_params_를 통해서 나온 값을 투입
rf =RandomForestRegressor(parameters, cv=kfold, verbose=2,random_state=1217)

rf.fit(X_train,y_train) #학습 

test['18~20_ride'] = rf.predict(X_test) #예측값 생성 후, test['18~20_ride']에 집어 넣는다.

y_predict=test['18~20_ride']

#100번 돈다.
print("최적의 매개변수",rf.best_estimator_) #모델 최고의 평가자

print("최종정답률",accuracy_score(y_test,y_predict))

# test[['id','18~20_ride']].to_csv("./mini_project/model/middle2.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 내보낸다

