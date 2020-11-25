#기준 xgboost 불러오기 
#1. 0 인놈 제거
#2. 하위 30% 제거
#3. 디폴드와 성능 비교
#winequality-white.csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

wine=pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)

y=wine['quality']
x=wine.drop('quality', axis=1)

print(x.shape) #(4898, 11)
print(y.shape) #(4898,)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("acc :", model.score(x_test, y_test))
print(model.feature_importances_)


import matplotlib.pyplot as plt
import numpy as np

plt.show()