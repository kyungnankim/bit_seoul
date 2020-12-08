import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np #데이터 처리
import pandas as pd #데이터 처리
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, r2_score
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
from xgboost import XGBClassifier, XGBRegressor
train_target=np.load('./mini_project/data/train_target.npy', allow_pickle=True)
train_data=np.load('./mini_project/data/train_data.npy', allow_pickle=True)

test_target=np.load('./mini_project/data/test_target.npy', allow_pickle=True)
test_data=np.load('./mini_project/data/test_data.npy', allow_pickle=True)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_target, train_data, train_size=0.8, random_state=66,shuffle=True)

from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score

# Create the parameter grid based on the results of random search 
params = [
    {"nini__n_estimators":[100,200,300], "nini__learning_rate":[0.1,0.3,0.01,0.001], 
    "nini__max_depth":[4,5,6]},
    {"nini__n_estimators":[90,100,110], "nini__learning_rate":[0.1,0.01,0.001], 
    "nini__max_depth":[4,5,6], "nini__colsample_bytree":[0.6,0.9,1]},
    {"nini__n_estimators":[9,110], "nini__learning_rate":[0.1,0.001,0.5], 
    "nini__max_depth":[4,5,6], "nini__colsample_bytree":[0.6,0.9,1], 
    "nini__colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1

pipe = Pipeline([("scaler", MaxAbsScaler()),('nini', XGBRegressor())])

# Create a based model
rf = RandomizedSearchCV(pipe,params,verbose=2,cv=5,random_state=1217)

rf.fit(x_train,y_train) #학습 

y_test = rf.predict(x_test) #예측값 생성 후, test['18~20_ride']에 집어 넣는다.

y_predict = rf.predict(x_test)
print('r2_score:',r2_score(y_test, y_predict))
y_predictt = pd.DataFrame(y_test,y_predict)
 
plt.figure(figsize=(24,5))

# Ridge의 Coef를 barplot으로 그린다.
plt.bar( train_target.drop(drop_cols,1).columns,  lasso.coef_ )

# y=0인 horizental한 선을 그린다.
plt.axhline(y=0, color='r', linestyle='-')

plt.xticks(rotation=45)
plt.title("강수량에 따른 승차 차이");


y_predictt.to_csv("./mini_project/model/18~20.csv",index=False)
#Public : 2.630561926

# import matplotlib.pyplot as plt
# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()


# plt.plot(y_predict)
# plt.plot(y_predictt)

# plt.title('loss & acc')
# plt.ylabel('loss, acc')
# plt.xlabel('epoch')

# plt.legend(['y_predict', 'y_predictt'])
# plt.show()
