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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')

#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀

#1.데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

print(x.shape,y.shape)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=66)
parameters = [
     {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
     {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]},
     {"C": [1, 10, 100, 1000], "kernel": ["sigmoid"],"gamma" : [0.001,0.0001]}
]
kfold = KFold(n_splits=5, shuffle=True)
# model = SVC()
model = GridSearchCV(SVC(),parameters, cv=kfold)
                        #cv =cross validation
model.fit(x_train, y_train)
#100번 돈다.
print("최적의 매개변수",model.best_estimator_) #모델 최고의 평가자

y_predict = model.predict(x_test)
print("최종정답률",accuracy_score(y_test,y_predict))

'''
(150, 4) (150,)
최적의 매개변수 SVC(C=1, kernel='linear')
최종정답률 1.0
'''