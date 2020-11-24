#당뇨병 RandomForestClassifier
#보스톤 RandomForestRegressor
#와인 RandomForestClassifier
#파일을 gridesearch 3, 4 ,5

#유방암 데이터 2진분류

#분류 select 모델 11
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.datasets import load_diabetes,load_boston,load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')

#RandomForest : 상당히 중요하다! 상당히 중요하다!!!!!!!!!!!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀

#1.데이터
diabetes_x, diabetes_y= load_diabetes(return_X_y=True)
boston_x, boston_y= load_boston(return_X_y=True)
wine_x, wine_y= load_wine(return_X_y=True)

diabetes_x_train, diabetes_x_test, diabetes_y_train, diabetes_y_test=train_test_split(diabetes_x, diabetes_y, test_size=0.2, random_state=66)
boston_x_train, boston_x_test, boston_y_train, boston_y_test=train_test_split(boston_x, boston_y, test_size=0.2, random_state=66)
wine_x_train, wine_x_test, wine_y_train, wine_y_test=train_test_split(wine_x, wine_y, test_size=0.2, random_state=66)

# 2. 모델
'''
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor
digits = load_boston()
X, y = dataset.data, dataset.target
model = RandomForestRegressor(random_state=30)
param_grid = { "n_estimators"      : [250, 300],
              "criterion"         : ["gini", "entropy"],
              "max_features"      : [3, 5],
              "max_depth"         : [10, 20],
              "min_samples_split" : [2, 4] ,
              "bootstrap": [True, False]}
grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=2)
grid_search.fit(X, y)
print grid_search.best_params_
'''
parameters= [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1]}
]

kfold = KFold(n_splits=5, shuffle=True)

# model = SVC()
model_diabetes = GridSearchCV(RandomForestClassifier(),parameters, cv=kfold)
model_boston = GridSearchCV(RandomForestRegressor(),parameters, cv=kfold)
model_wine = GridSearchCV(RandomForestClassifier(),parameters, cv=kfold)
                        #cv =cross validation
model_diabetes.fit(diabetes_x_train, diabetes_y_train)
model_boston.fit(boston_x_train, boston_y_train)
model_wine.fit(wine_x_train, wine_y_train)

#100번 돈다.
print("model_diabetes 최적의 매개변수",model_diabetes.best_estimator_) #모델 최고의 평가자
print("model_boston 최적의 매개변수",model_boston.best_estimator_) #모델 최고의 평가자
print("model_wine 최적의 매개변수",model_wine.best_estimator_) #모델 최고의 평가자

diabetes_y_predict = model_diabetes.predict(diabetes_x_test)
print("최종정답률",accuracy_score(diabetes_y_test,diabetes_y_predict))

wine_y_predict = model_wine.predict(wine_x_test)
print("최종정답률",accuracy_score(wine_y_test,wine_y_predict))


#####4. 평가, 예측
score = model_boston.score(boston_x_test, boston_y_test)
print("model.score : ", score)

#accuracy_score를 넣어서 비교할 것
#회귀모델일 경우  r2_score로 코딩해서 score와 비교할 것
boston_y_predict=model_boston.predict(boston_x_test)
# acc=accuracy_score(y_test, y_predict)
r2=r2_score(boston_y_test, boston_y_predict)

print('score :', score)
# print('acc :', acc)
print('r2 : ', r2)

print(boston_y_test[:10],'의 예측결과','\n',boston_y_predict)
# # 실습, 결과물 오차 수집, 미세조절
'''


<장단점과 매개변수>
회귀와 분류에 있어서 랜덤 포레스트는 현재 가장 널리 사용되는 머신러닝 알고리즘이다.
랜덤 포레스트는 성능이 매우 뛰어나고 매개변수 튜닝을 많이 하지 않아도 잘 작동하며, 데이터의 스케일을 맞출 필요도 없다.
기본적으로 랜덤 포레스트는 단일 트리의 단점을 보완하고 장점은 가지고 있다.
대량의 데이터셋에서 랜덤 포레스트 모델을 만들 때 다소 시간이 걸릴 수 있지만 CPU코어가 많다면 손쉽게 병렬 처리할 수있다.
n_jobs 매개변수를 이용하여 사용할 코어 수를 지정할  수 있다.
(n_jobs=-1로 지정하면 컴퓨터의 모든 코어를 사용한다.)
주의할점
랜덤 포레스트는 랜덤하기때문에 random_state를 다르제 지정하면 전혀 다른 모델이 만들어진다.
당연히 랜덤 포레스트의 트리가 많을수록 random_state값의 변화에 따른 변동이 적다.
랜덤 포레스트는 텍스트 데이터와 같이 매우 차원이 놓고 희소한 데이터에는 잘 작동하지 않는다.
이러한 데이터에는 선형 모델이 더 적합하다.
메모리를 많이 사용하기에 훈련과 예측이 느리다.
중요 매개변수는 n_estimators, max_features이고 max_depth 같은 사전 가지치기 옵션이 있다.
n_estimators는 클수록 좋다. 더 많은 트리를 평군하면 과대 적합을 줄여 안정적인 모델을 만둘 수 있다. (메모리와 훈련시간은 증가한다.)
max_features는 각 트리가 얼마나 무작위가 될지를 결정하며, 작은 max_features는 과대적합을 줄인다. 일반적으로는 기본값을 쓰면된다.
'''