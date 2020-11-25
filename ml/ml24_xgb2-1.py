#과적합방지
#1. 혼란 데이터 량을 늘린다.
#2. v피쳐수를 줄인다.
#3. regularfraion
from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

dataset=load_boston()
x=dataset.data
y=dataset.target

pca1=PCA(n_components=12)
x=pca1.fit_transform(x)

learning_rate=0.01
n_estimators=100
colsample_bylevel=1
colsample_bytree=1

max_depth=list(range(2,10))
n_jobs=8

model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
                     n_estimators=n_estimators, n_jobs=n_jobs,
                     colsample_bylevel=colsample_bylevel,
                     colsample_bytree=colsample_bytree)

#score  디폴트 했던 놈과 성능 비교

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

model=XGBClassifier(max_depth=6)
model.fit(x_train, y_train)

y_predict=model.predict(x_test)

print("R2 :", r2_score(y_test, y_predict))
print(model.feature_importances_)


plt.plot(x) # x값들이 점으로 분포가 되어야되는데 
plt.plot(y)

plt.show()