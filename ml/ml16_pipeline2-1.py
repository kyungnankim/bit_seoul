#분류 select 모델 11
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
#1.데이터
x, y=load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True)
parameters = [
    {"maldding__C":[1,10,100,1000],"maldding__kernel":["linear"]}, # 4x1번
    {"maldding__C":[1,10,100,1000],"maldding__kernel":["rbf"],"maldding__gamma":[0.001,0.0001]}, # 4x1x2번
    {"maldding__C":[1,10,100,1000],"maldding__kernel":["sigmoid"],"maldding__gamma":[0.001,0.0001]} # 4x1x2번
] # 총 4+8+8=20번
#2.모델
pipe = Pipeline([("scaler", MaxAbsScaler()), ('maldding',SVC())])

model = RandomizedSearchCV(pipe, parameters, cv=5)
model.fit(x_train, y_train)
print('acc : ', model.score(x_test, y_test)) #(150, 4) (150,) acc :  1.0
# print('최적의 매개변수 : ', model.best_estimator_) #(150, 4) (150,) acc :  1.0
# print('최적의 매개변수 : ', model.best_params_) #(150, 4) (150,) acc :  1.0

