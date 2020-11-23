#다중분류
import numpy as np
from sklearn.datasets import load_iris
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
x, y= load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66,shuffle=True, train_size=0.8)

scale=StandardScaler()
scale.fit(x_train)
x_train=scale.transform(x_train)
x_test=scale.transform(x_test)

#####2. 모델
# model = LinearSVC()
model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

####3. 훈련
model.fit(x_train, y_train)

#####4. 평가, 예측
score = model.score(x_test, y_test)
print("model.score : ", score)

#accuracy_score를 넣어서 비교할 것
#회귀모델일 경우  r2_score로 코딩해서 score와 비교할 것
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuract_Score : ", acc)

print(y_test[:10],'의 예측결과','\n',y_predict)
# 실습, 결과물 오차 수집, 미세조절

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)
''' 이게 무조건 좋은게 아니라 iris_dataset 에 대해서 좋은 모델이지 무조건 좋은 모델이 아니다.!
model = LinearSVC()
model.score :  0.9333333333333333
accuract_Score :  0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측결과
 [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 1 1 2]

model = SVC()
model.score :  0.9333333333333333
accuract_Score :  0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측결과
 [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 1 2 1 2 0 1 2 2]

model = KNeighborsClassifier()
model.score :  0.9
accuract_Score :  0.9
[1 1 1 0 1 1 0 0 0 2] 의 예측결과
 [1 1 2 0 1 1 0 0 0 2 2 2 0 1 2 0 1 2 2 2 0 1 1 2 1 2 0 0 2 2]

model = KNeighborsRegressor()
model.score :  0.8191516065589246
RMSE :  0.20221
R2 :  0.8191516065589246

model = RandomForestClassifier()
model.score :  0.9333333333333333
accuract_Score :  0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측결과
 [1 1 1 0 1 1 0 0 0 2 2 2 0 1 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]

model = RandomForestRegressor()
model.score :  0.9333333333333333
accuract_Score :  0.9333333333333333
[1 1 1 0 1 1 0 0 0 2] 의 예측결과
 [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 1 2 1 2 0 1 2 2]
RMSE :  0.2581988897471611
R2 :  0.8993288590604027
'''