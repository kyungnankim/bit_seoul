import pandas as pd
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)

count_data = wine.groupby('quality')['quality'].count()

y = wine['quality']
x = wine.drop('quality',axis=1)

print(x.shape)#(4898, 11)
print(y.shape)#(4898, )

newlist = []
for i in list(y):
    if i <= 4:
      newlist +=[0]
    elif i <= 7:
      newlist +=[1]
    else :
       newlist +=[2]

y = newlist

# 모델 만든거 이어라
print(x.shape)
# print(y.shape)

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
model = RandomForestClassifier()
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