#당뇨병 RandomForestClassifier
#보스톤 RandomForestRegressor
#와인 RandomForestClassifier
#파일을 gridesearch 3, 4 ,5

#분류 select 모델 11
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils.testing import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
#GridSearchCV → RandomizedSearchCV 모델부터 바꿔야함. 어렵다. 주의할 것
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
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
parameters= [
    {'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,10],
    'min_samples_split' : [2,3,5,10],
    'n_jobs' : [-1]}
]
kfold = KFold(n_splits=5, shuffle=True)
model=GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=2) # 총 640번 훈련


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', accuracy_score(y_test, y_predict))

'''
(150, 4) (150,)
최적의 매개변수 SVC(C=1, kernel='linear')
최종정답률 1.0

(150, 4) (150,)
최적의 매개변수 SVC(C=1000, gamma=0.0001)
최종정답률 1.0
'''