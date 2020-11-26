#실습
#1. 상단 모델에 그리스 서치 또는 랜덤 서치 적용
#최적의  R2값과 피처임포턴츠 구할 것

#2. 위 쓰레드 값으로 SelecFromModel 을 구해서 최적의 피쳐 갯수를 구할 것
#최적의 피처 갯수를 구할 것
#위 피쳐 갯수로 데이터(피처)를 수정(삭제)
#그리드서치 또는 랜덤서치 적용해서 최적의 R2값을 구할 것

#1번 값과 2번 값을 비교해 볼 것
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
# 1.
'''
x, y=load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

params = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.01,0.001], 
    "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[9,110], "learning_rate":[0.1,0.001,0.5], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], 
    "colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1
#2. 모델
kfold=KFold(n_splits=5, shuffle=True)
model=RandomizedSearchCV(XGBRegressor(), params, cv=kfold) 

#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('feature_importance : ', XGBRegressor().fit(x_train, y_train).feature_importances_)


# 2.
x, y=load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model = XGBRegressor(max_depth=6, learning_rate=0.01,
                     n_estimators=100, n_jobs=8,
                     colsample_bylevel=1,
                     colsample_bytree=1)

model.fit(x_train, y_train)
score=model.score(x_test, y_test)

print('score :', score)

thresholds=np.sort(model.feature_importances_)
print('feature_importance SORT : ',thresholds)

for thresh in thresholds :
    selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train=selection.transform(x_train)
    
    selection_model=XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    selec_x_test=selection.transform(x_test)
    y_predict=selection_model.predict(selec_x_test)

    score=r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
'''


# 3.
x, y=load_boston(return_X_y=True)
x_data1=x[:,:4]
x_data2=x[:,5:]
x=np.concatenate([x_data1, x_data2], axis=1)
print(x.shape) #(506, 12)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.01,0.001], 
    "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[9,110], "learning_rate":[0.1,0.001,0.5], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], 
    "colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1
#2. 모델
kfold=KFold(n_splits=5, shuffle=True)
model=RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold)


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('feature_importance : ', XGBRegressor().fit(x_train, y_train).feature_importances_)
'''
(506, 12)
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=110, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
최종정답률 :  0.8515198650026703
feature_importance :  [0.01608834 0.00282385 0.01047728 0.00192173 0.26891148 0.00889584
 0.03910455 0.01072508 0.04006825 0.06881987 0.01746926 0.5146944 ]

'''