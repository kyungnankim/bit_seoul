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
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66,shuffle=True, train_size=0.8)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)

#####2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
model = RandomForestRegressor()

####3. 훈련
model.fit(x_train, y_train)

#####4. 평가, 예측
score = model.score(x_test, y_test)
print("model.score : ", score)

#accuracy_score를 넣어서 비교할 것
#회귀모델일 경우  r2_score로 코딩해서 score와 비교할 것
y_predict=model.predict(x_test)
# acc=accuracy_score(y_test, y_predict)
r2=r2_score(y_test, y_predict)

print('score :', score)
# print('acc :', acc)
print('r2 : ', r2)

print(y_test[:10],'의 예측결과','\n',y_predict)
# # 실습, 결과물 오차 수집, 미세조절

''' 이게 무조건 좋은게 아니라 iris_dataset 에 대해서 좋은 모델이지 무조건 좋은 모델이 아니다.!
model = LinearSVC()
ValueError: Unknown label type: 'continuous'

model = SVC()
ValueError: Unknown label type: 'continuous'

model = KNeighborsClassifier()
ValueError: Unknown label type: 'continuous'

model = KNeighborsRegressor()
ValueError: Unknown label type: 'continuous'

model = RandomForestClassifier()


model = RandomForestRegressor()
model.score :  0.9161791578553041
score : 0.9161791578553041
r2 :  0.9161791578553041
[16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1] 의 예측결과
 [15.195 46.42  27.872 45.056 21.078 21.117 20.002 20.642 45.197 16.497
 21.529 11.925 34.28  23.469 15.787 20.312 14.745 15.243  7.113 14.829
 20.783 19.355 21.374 41.651 18.641 26.498 21.323 19.789 15.548 29.473
 23.349 29.966 44.914 15.936 26.893 23.094 26.856 23.983 19.651 21.423
 23.632 21.135 17.797 48.247 30.767 18.675 22.914 12.765 20.85  21.805
 31.409 17.722 45.244 23.838 13.842 20.078 23.855 20.102 20.365 20.198
 20.324 20.086 10.015 22.017 20.907 11.731 19.716 25.874 21.9   20.754
  8.492 22.113 22.193 15.157 21.01  30.567 30.691  8.145 19.659  7.845
 30.87  11.757 26.588 22.883 22.761 30.935 31.363 21.23  21.151 35.437
 20.899 47.849 18.519 21.803 20.778 13.713 20.222 24.871 22.928 21.79
 20.418 18.457]
'''