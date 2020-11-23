#다중분류
import numpy as np
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#RandomForest : 상당히 중요하다!

#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀
#1.데이터
x, y= load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66,shuffle=True, train_size=0.8)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)
# 2. 모델
# model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=KNeighborsRegressor()
model=RandomForestClassifier(n_estimators=100, n_jobs=-1)
# model=RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score=model.score(x_test, y_test)

# accuracy_score를 넣어서 비교할 것
# 회귀모델일 경우 r2_score와 비교할 것

y_predict=model.predict(x_test)
acc=accuracy_score(y_test, y_predict)
# r2=r2_score(y_test, y_predict)

print('score :', score)
print('acc :', acc)
# print('r2 : ', r2)

print(y_test[:10], '의 예측 결과 ', y_predict[:10])

''' 이게 무조건 좋은게 아니라 iris_dataset 에 대해서 좋은 모델이지 무조건 좋은 모델이 아니다.!
model = LinearSVC()
score : 0.9736842105263158
acc : 0.9736842105263158
[1 1 1 1 1 0 0 1 1 1] 의 예측 결과  [1 1 1 1 1 0 0 1 1 1]

model = SVC()
score : 0.9649122807017544
acc : 0.9649122807017544
[1 1 1 1 1 0 0 1 1 1] 의 예측 결과  [1 1 1 1 1 0 0 1 1 1]

model = KNeighborsClassifier()
score : 0.956140350877193
acc : 0.956140350877193
[1 1 1 1 1 0 0 1 1 1] 의 예측 결과  [1 1 1 1 1 0 0 1 1 1]

model = KNeighborsRegressor()
ValueError: Classification metrics can't handle a mix of binary and continuous targets

model = RandomForestClassifier()
score : 0.956140350877193
acc : 0.956140350877193
[1 1 1 1 1 0 0 1 1 1] 의 예측 결과  [1 1 1 1 1 0 0 1 1 1]

model = RandomForestRegressor()
Classification metrics can't handle a mix of binary and continuous targets
'''