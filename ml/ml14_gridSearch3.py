#당뇨병

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

x, y=load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

'''
parameters= [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1]}
]
'''
parameters= [
    {'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    "max_features"      : [3, 5],
    'min_samples_leaf' : [3,5,7,10],
    'min_samples_split' : [2,3,5,10],
    'n_jobs' : [-1]}
] #128번

#2. 모델
kfold=KFold(n_splits=5, shuffle=True) # 5번
model=GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=2) # 총 640번 훈련


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))

