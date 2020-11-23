# winequality-white.csv

#다중분류
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#RandomForest : 상당히 중요하다!
winequality = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=0, encoding='CP949',sep=',' )
print(winequality)
print(winequality.shape) (4898, 0)
print(winequality.describe())
# winequality_npy = winequality.values

# x_train, x_test, y_train, y_test = train_test_split(
#    winequality, random_state=66,shuffle=True, train_size=0.8)

# x = winequality_npy[:,0:11]
# y = winequality_npy[:,11]

# scale=StandardScaler()
# scale.fit(x_train)
# x_train=scale.transform(x_train)
# x_test=scale.transform(x_test)
# x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# #####2. 모델
# # model = LinearSVC()
# # model = SVC()
# # model = KNeighborsClassifier()
# # model = KNeighborsRegressor()
# model = RandomForestClassifier()
# # model = RandomForestRegressor()

# ####3. 훈련
# model.fit(x_train, y_train)

# #####4. 평가, 예측
# score = model.score(x_test, y_test)
# print("model.score : ", score)


    
# np.save('./data/csv/winequality-white.npy', arr=winequality)