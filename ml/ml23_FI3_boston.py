#pca 축소해서 모델을 완성하세요.
#1. 0.95  이상
#2. 1이상
#minist dnn 과 loss/acc 를 비교

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from sklearn.datasets import load_iris, load_diabetes,load_boston,load_breast_cancer,load_wine

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score
from matplotlib import pyplot as plt
from xgboost import XGBClassifier

dataset=load_boston()
x=dataset.data
y=dataset.target

pca1=PCA(n_components=12)
x=pca1.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("R2 :", r2_score(y_test, y_predict))
print(model.feature_importances_)
