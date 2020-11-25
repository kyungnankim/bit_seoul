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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀

#1.데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1]

print(x.shape,y.shape)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True)
pipe = Pipeline([("scaler", MaxAbsScaler()), ('maldding',SVC())])
'''
x = iris.iloc[:, :4]
y = iris.iloc[:, -1]
StandardScaler
(150, 4) (150,)
acc :  0.9666666666666667
MinMaxScaler
(150, 4) (150,)
acc :  1.0
RobustScaler
(150, 4) (150,)
acc :  0.9666666666666667
MaxAbsScaler
(150, 4) (150,)
acc :  1.0
'''
pipe.fit(x_train, y_train)
print('acc : ', pipe.score(x_test, y_test)) #(150, 4) (150,) acc :  1.0
