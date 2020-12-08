#데이터 처리
import numpy as np 
import pandas as pd

#데이터 샘플링
import random
from sklearn.preprocessing import LabelEncoder #인코딩
from sklearn.preprocessing import OneHotEncoder

#validation
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV #모델링
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
#model
import lightgbm as lgbm 


train=np.load('./mini_project/data/train_nup.npy', allow_pickle=True)
test=np.load('./mini_project/data/test_nup.npy', allow_pickle=True)
train = train.astype(np.int64)
test = test.astype(np.int64)
print(type(train))
print(type(test))
def erase(data):
    for i in range(len(data.string)):
        for j in range(len(data.iloc[i])):
            data.iloc[i,j]=int(data.iloc[i,j].replace(',',''))
            # print("i,j:",i,"/",j)
    return data
train = erase(train)

input_var=[['68a', '810a', '1012a', '68b', '810b', '1012b', 'precipitation']]

target=train(['18~20_ride'])

X_train=train[input_var]
random.seed(1217) #동일한 샘플링하기 위한 시드번호
train_list=random.sample(list(range(X_train.shape[0])), int(round(X_train.shape[0]*0.01,0)) )

X_train=train[input_var]
X_train=X_train.iloc[train_list,:]
y_train=train[target]
y_train=y_train.iloc[train_list,:]

X_test=test2[input_var]

X_train.shape, y_train.shape

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

grid_search.best_params_

#전체 데이터로 적용
X_train=train[input_var]
y_train=train[target]

X_test=test2[input_var]

X_train.shape, y_train.shape, X_test.shape
((415423, 26), (415423, 1), (228170, 26))

# best_params_를 통해서 나온 값을 투입
rf = RandomForestRegressor(max_features=3,min_samples_leaf=2,min_samples_split=2,n_estimators=500,random_state=1217)

rf.fit(X_train,y_train) #학습 

test['18~20_ride'] = rf.predict(X_test) #예측값 생성 후, test['18~20_ride']에 집어 넣는다.

test[['id','18~20_ride']].to_csv("dacon_base_middle2.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 내보낸다

#해당 코드 소요 시간 5분