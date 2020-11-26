import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import all_estimators
#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀

#1.데이터
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

print(x.shape, y.shape) #(150, 4) (150, 5)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66,shuffle=True, train_size=0.8)

#####2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
kfold = KFold(n_splits=5,shuffle=True)
'''
model_0 = LinearSVC()
scores_0 = cross_val_score(model_0, x_train, y_train, cv=kfold)
        #검증한 score 
        # model.score 분류일 때는 acc_score, 회귀일 때는 r2_score
print('scores_0',scores_0)

model_1 = SVC()
scores_1 = cross_val_score(model_1, x_train, y_train, cv=kfold)
        #검증한 score 
        # model.score 분류일 때는 acc_score, 회귀일 때는 r2_score
print('scores_1',scores_1)

model_2 = KNeighborsClassifier()
scores_2 = cross_val_score(model_2, x_train, y_train, cv=kfold)
        #검증한 score 
        # model.score 분류일 때는 acc_score, 회귀일 때는 r2_score
print('scores_2',scores_2)


model_3 = KNeighborsRegressor()
scores_3 = cross_val_score(model_3, x_train, y_train, cv=kfold)
        #검증한 score 
        # model.score 분류일 때는 acc_score, 회귀일 때는 r2_score
print('scores_3',scores_3)

model_4 = RandomForestClassifier()
scores_4 = cross_val_score(model_4, x_train, y_train, cv=kfold)
        #검증한 score 
        # model.score 분류일 때는 acc_score, 회귀일 때는 r2_score
print('scores_4',scores_4)

model_5 = RandomForestRegressor()
scores_5 = cross_val_score(model_5, x_train, y_train, cv=kfold)
        #검증한 score 
        # model.score 분류일 때는 acc_score, 회귀일 때는 r2_score
print('scores_5',scores_5)
'''
kfold=KFold(n_splits=5, shuffle=True)
# classifier, regressor
allAlgorithms_1 = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms_1 :
    try :
        model=algorithm()
        scores=cross_val_score(model, x_train, y_train, cv=kfold)#나중에는 kfold를 따로 정의하지 않고 cv에 숫자만 넣어도 가능하다.
        print('allAlgorithms_1 =',name, '의 정답률 : ', scores)
    except :
        pass


allAlgorithms_2 = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms_2 :
    try :
        model=algorithm()
        scores=cross_val_score(model, x_train, y_train, cv=kfold)#나중에는 kfold를 따로 정의하지 않고 cv에 숫자만 넣어도 가능하다.
        print('allAlgorithms_2 =', name, '의 정답률 : ', scores)
    except :
        pass


allAlgorithms_3 = all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms_3 :
    try :
        model=algorithm()
        scores=cross_val_score(model, x_train, y_train, cv=5)#나중에는 kfold를 따로 정의하지 않고 cv에 숫자만 넣어도 가능하다.
        print('allAlgorithms_3 =', name, '의 정답률 : ', scores)
    except :
        pass
'''
scores_0 [0.875      0.83333333 0.91666667 0.75       0.66666667]
scores_1 [1.         0.83333333 1.         1.         1.        ]
scores_2 [0.95833333 0.95833333 1.         1.         1.        ]
scores_3 [0.98732673 0.985      0.95358242 0.96523677 0.99764128]
scores_4 [1.         1.         1.         1.         0.95833333]
scores_5 [0.98699652 0.99999373 0.99826367 0.9658752  0.9984972 ]
'''
