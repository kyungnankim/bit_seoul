import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier, XGBRegressor, plot_importance

x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

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

model = GridSearchCV(XGBRegressor(), params, n_jobs=n_jobs, cv=kfold, verbose=2)

model.fit(x_train,y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

r2 = model.score(x_test,y_test)
print("r2 : ", r2)
