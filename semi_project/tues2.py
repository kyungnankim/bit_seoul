import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd #데이터 처리
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from lightgbm import LGBMRegressor, LGBMClassifier, plot_importance
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import lightgbm
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import OneHotEncoder
train=np.load('./mini_project/data/train_nupp.npy', allow_pickle=True)
test=np.load('./mini_project/data/train_nupp.npy', allow_pickle=True)

train = train.astype(int)
print(type(train))
# test = test.astype(np.int64)
# print("x train shape : ", train.shape) #(1061878, 67)
# print("y train shape : ", test.shape) #(310121, 67)

# print(train.shape)
# print(test.shape)
# train = train.T
# test = test.T

# train = train[:5,:]
# test = test[:5]

# x_train, x_test, y_train, y_test = train_test_split(test, train, shuffle=True, train_size=0.8, random_state=77)

# # 각 모델에 대한 oof 정의
# lgbm_oof_train = np.zeros((train.shape[0]))
# lgbm_oof_test = np.zeros((test.shape[0]))

# # Hyperparameter 정의하기
# n_splits= 5
# NUM_BOOST_ROUND = 10000
# SEED = 777
# lgbm_param = {'objective':'rmse',
#               'boosting_type': 'gbdt',
#               'random_state':777,
#               'learning_rate':0.1,
#               'subsample':0.7,
#               'tree_learner': 'serial',
#               'colsample_bytree':0.78,
#               'early_stopping_rounds':50,
#               'subsample_freq': 1,
#               'reg_lambda':7,
#               'reg_alpha': 5,
#               'num_leaves': 96,
#               'seed' : SEED
#             }

# # Kfold 
# kfolds = KFold(n_splits=n_splits, random_state=777, shuffle=True)
# target_col = train['18~20_ride']
# train_label = train[target_col]
# # Fold별로 학습진행
# for ind, (trn_ind, val_ind) in tqdm(enumerate( kfolds.split( X = train, y = train_label ) ) ):
    
#     # Train/Valid-set을 정의
#     X_train , y_train = train.iloc[trn_ind], train_label[trn_ind]
#     X_valid , y_valid = train.iloc[val_ind], train_label[val_ind]
    
#     # Light GBM
#     print("---TRAINING---")
    
#     # dtrain/dvalid 정의
#     dtrain = lgbm.Dataset(X_train, y_train)
#     dvalid = lgbm.Dataset(X_valid, y_valid)
    
#     # model 정의&학습
#     model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 
#                        valid_sets=(dtrain, dvalid), 
#                        valid_names=('train','valid'), 
#                        verbose_eval= 100)
    
#     # local_valid/local_test에 대한 예측
#     lgbm_valid_pred = model.predict(X_valid)
#     lgbm_test_pred = model.predict(test)
        
#     lgbm_oof_train[val_ind] = lgbm_valid_pred
#     lgbm_oof_test += lgbm_test_pred/ n_splits #Fold한 결과 앙상블(평균)

# from math import sqrt
# from sklearn.metrics import mean_squared_error

# print(f"<Light-GBM> OVERALL RMSE: {sqrt( mean_squared_error( train_label, lgbm_oof_train ))}")
# #<Light-GBM> OVERALL RMSE     : 2.4589149460117166

# X_train=train
# y_train=train_label
# X_test=test

# sample_submission = pd.read_csv('./mini_project/data/submission_sample.csv')

# sample_submission[target_col] = model.predict(X_test)

# sample_submission.to_csv("./mini_project/model/submission.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 저장.

# #Public : 2.630561926