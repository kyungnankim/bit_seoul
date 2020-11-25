import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

print(x.shape)

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=4)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("acc :", model.score(x_test, y_test))
print(model.feature_importances_)