import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score

#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀

#1.데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

print(x.shape, y.shape) #(150, 4) (150, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66,shuffle=True, train_size=0.8)

#####2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
kfold = KFold(n_splits=5,shuffle=True)

model = SVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
        #검증한 score 
        # model.score 분류일 때는 acc_score, 회귀일 때는 r2_score
print('scores',scores)

'''
(150, 4) (150,)
scores [1.         1.         0.875      1.         0.95833333]
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
