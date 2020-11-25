#기준 xgboost 불러오기 
#1. 0 인놈 제거
#2. 하위 30% 제거
#3. 디폴드와 성능 비교

#Tree구조의 모델들 성능이 다른 모델에 비해 좋음. keras보다 잘 나올 수도 있음.
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np

dataset=load_diabetes()
x=dataset.data
y=dataset.target

pca1=PCA(n_components=7)
x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("score :", model.score(x_test, y_test))
print(model.feature_importances_)
