#분류 select 모델 11
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀
from sklearn.datasets import load_boston

#1.데이터
x, y=load_boston(return_X_y=True)

print(x.shape,y.shape)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True)
parameters= [
    {'nini__n_estimators' : [100,200],
    'nini__max_depth' : [6,8,10,12],
    'nini__max_features'  : [7, 9],
    'nini__min_samples_leaf' : [3,5,7,10],
    'nini__min_samples_split' : [2,3,5,10],
    'nini__n_jobs' : [-1]}
]


#2. 모델
pipe=Pipeline([("scaler", MinMaxScaler()), ('nini', RandomForestRegressor())])
# pipe=make_pipeline(MinMaxScaler(), RandomForestRegressor()) 
model=RandomizedSearchCV(pipe, parameters, cv=5) 


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('최적의 매개변수 : ', model.best_estimator_) #(150, 4) (150,) acc :  1.0
print('최적의 매개변수 : ', model.best_params_) #(150, 4) (150,) acc :  1.0


'''
(506, 13) (506,)
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('nini',
                 RandomForestRegressor(max_depth=6, min_samples_leaf=3,
                                       min_samples_split=3, n_jobs=-1))])
최종정답률 :  0.922036088351571
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('nini',
                 RandomForestRegressor(max_depth=6, min_samples_leaf=3,
                                       min_samples_split=3, n_jobs=-1))])
최적의 매개변수 :  {'nini__n_jobs': -1, 'nini__n_estimators': 100, 'nini__min_samples_split': 3, 'nini__min_samples_leaf': 3, 'nini__max_depth': 6}
'''