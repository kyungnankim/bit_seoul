import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score

#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀

#1.데이터
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
scores_0 [0.875      0.83333333 0.91666667 0.75       0.66666667]
scores_1 [1.         0.83333333 1.         1.         1.        ]
scores_2 [0.95833333 0.95833333 1.         1.         1.        ]
scores_3 [0.98732673 0.985      0.95358242 0.96523677 0.99764128]
scores_4 [1.         1.         1.         1.         0.95833333]
scores_5 [0.98699652 0.99999373 0.99826367 0.9658752  0.9984972 ]
'''


# ####3. 훈련
# model.fit(x_train, y_train)

# # 4. 평가, 예측
# from sklearn.metrics import accuracy_score, r2_score
# score=model.score(x_test, y_test)

# # accuracy_score를 넣어서 비교할 것
# # 회귀모델일 경우 r2_score와 비교할 것

# y_predict=model.predict(x_test)
# acc=accuracy_score(y_test, y_predict)
# # r2=r2_score(y_test, y_predict)

# print('score :', score)
# print('acc :', acc)
# # print('r2 : ', r2)

# print(y_test[:10], '의 예측 결과 ', y_predict[:10])
