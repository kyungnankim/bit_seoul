import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np #데이터 처리
import pandas as pd #데이터 처리
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import lightgbm
import random
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC, SVC
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier, XGBRegressor


x=np.load('./mini_project/data/train_target.npy')#, allow_pickle=True
y=np.load('./mini_project/data/train_data.npy')#, allow_pickle=True

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66,shuffle=True)
# train_target = train_target.astype('int64')
# train_data = train_data.astype('int64')
# test_target = test_target.astype('int64')
# test_data = test_data.astype('int64')
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

model = XGBRegressor()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)


print(model.feature_importances_)



# 시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance_cancer(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features),x)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importance_cancer(model)
plt.show()
