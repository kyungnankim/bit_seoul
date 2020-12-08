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
train_target=np.load('./mini_project/data/train_target.npy', allow_pickle=True)
train_data=np.load('./mini_project/data/train_data.npy', allow_pickle=True)

test_target=np.load('./mini_project/data/test_target.npy', allow_pickle=True)
test_data=np.load('./mini_project/data/test_data.npy', allow_pickle=True)


print(train_target.shape)
print(train_data.shape)
print(test_target.shape)
print(test_data.shape)

# 각 모델에 대한 oof 정의
train_label = train_data

# Hyperparameter 정의하기
n_splits= 5
NUM_BOOST_ROUND = 10000
SEED = 888
lgbm_param = {'objective':'rmse', 'boosting_type': 'gbdt',
              'random_state':888, 'learning_rate':0.1,
              'subsample':0.7, 'tree_learner': 'serial',
              'colsample_bytree':0.78, 'early_stopping_rounds':50,
              'subsample_freq': 1,'reg_lambda':7,
              'reg_alpha': 5,'num_leaves': 96, 'seed' : SEED
}

# 각 모델에 대한 oof 정의
lgbm_oof_train = np.zeros((train_target.shape[0]))
lgbm_oof_test = np.zeros((test_target.shape[0]))
# Kfold 
kfolds = KFold(n_splits=n_splits, random_state=777, shuffle=True)
from lightgbm import plot_importance, Dataset, LGBMRegressor
from tqdm.notebook import tqdm

# Fold별로 학습진행
for ind, (trn_ind, val_ind) in tqdm(enumerate(kfolds.split( X = train_target, y = train_label))):
    # Train/Valid-set을 정의
    X_train , y_train = train_target[trn_ind], train_label[trn_ind]
    X_valid , y_valid = train_target[val_ind], train_label[val_ind]
    # Light GBM
    print("---TRAINING---")
    # dtrain/dvalid 정의
    lgbm = LGBMRegressor(n_estimators=1000, num_leaves=50, subsample=0.8, min_child_samples=60, max_depth=20)
    lgbm.fit(X_train,y_train)
    lgbm_valid_pred = lgbm.predict(X_valid)
    lgbm_test_pred = lgbm.predict(test_target)
    lgbm_oof_train[val_ind] = lgbm_valid_pred
    lgbm_oof_test += lgbm_test_pred/ n_splits
    
from math import sqrt
from sklearn.metrics import mean_squared_error
print(f"<Light-GBM> OVERALL RMSE: {sqrt( mean_squared_error( train_label, lgbm_oof_train ))}")
# <Light-GBM> OVERALL RMSE     : 2.4589149460117166
#4. 예측 및 결과값 저장
y_predict = lgbm.predict(test_target)
print(y_predict) #8.279504382440336

y_predictt = pd.DataFrame(y_predict)
 
y_predictt.to_csv("./mini_project/model/18~20.csv",index=False)
#Public : 2.630561926

import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

plt.plot(acc)
plt.plot(val_acc)
plt.plot(y_yloss)
plt.plot(y_loss)

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
plt.show()
