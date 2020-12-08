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

train_target=train_target[1000:10000,1:]
train_data=train_data[1000:10000,:]
test_target=test_target[1000:10000,:]
test_data=test_data[1000:10000,:]



# train_target = train_target.astype('int64')
# train_data = train_data.astype('int64')
# test_target = test_target.astype('int64')
# test_data = test_data.astype('int64')
print(train_target.shape)
print(train_data.shape)
print(test_target.shape)
print(test_data.shape)

# 각 모델에 대한 oof 정의
train_label = train_data

# Hyperparameter 정의하기
n_splits= 5
NUM_BOOST_ROUND = 10000
SEED = 777
lgbm_param = {'objective':'rmse', 'boosting_type': 'gbdt',
              'random_state':777, 'learning_rate':0.1,
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
from lightgbm import LGBMClassifier,plot_importance, Dataset
from tqdm.notebook import tqdm
# Fold별로 학습진행
for ind, (trn_ind, val_ind) in tqdm(enumerate(kfolds.split( X = train_target, y = train_label))):
    # Train/Valid-set을 정의
    X_train , y_train = train_target[trn_ind], train_label[trn_ind]
    X_valid , y_valid = test_target[val_ind], test_data[val_ind]
    # Light GBM
    print("---TRAINING---")
    # dtrain/dvalid 정의
    lgbm = LGBMClassifier(n_estimators=1000, num_leaves=50, subsample=0.8, min_child_samples=60, max_depth=20)
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
# y_predict.to_csv("./mini_project/model/18~20.csv",index=False)
#Public : 2.630561926































# x_train=train_data
'''
# train_list=random.sample(int(range(x_train.shape[0])), int(round(x_train.shape[0]*0.01,0)) )
y_test = test_data
x_train=train_target
y_train=train_data
# x_train=x_train.iloc[train_list,:]
# y_train=train_data.iloc[train_list,:]

x_test=test_target

param_grid = {
    'max_features': [2,3,5],
    'min_samples_leaf': [2,3],
    'min_samples_split': [2,4,6],
    'n_estimators': [100, 200,500]
}
# Create a based model
rf = RandomForestRegressor(random_state=1217) # 랜덤포레스트 모델을 정의한다.
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,verbose=2) # GridSearchCV를 정의한다.

grid_search.fit(x_train, y_train)

grid_search.best_params_ #학습 이후 최적의 paramter를 출력


# best_params_를 통해서 나온 값을 투입
rf = RandomForestRegressor(max_features=3,min_samples_leaf=2,min_samples_split=2,n_estimators=500,random_state=1217)

rf.fit(x_train,y_train) #학습 

y_predict = rf.predict(x_test) 
##########모델 검증

print(r2_score(y_test, y_predict)) #0.8526379440119077

##########모델 예측

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)
print(y_predict) #8.279504382440336

y_predict.to_csv("./mini_project/model/dacon_base_middle2.csv",index=False)
'''




















# x_train, x_test, y_train, y_test = train_test_split(train_target, train_data, test_target,train_size=0.5, random_state=SEED,stratify=train_data)
# x_train, x_temp, y_train, y_temp = train_test_split(train_target, train_data, test_data,train_size=0.5, random_state=SEED, stratify=train_data)
# x_test, x_temp, y_test, y_temp = train_test_split(test_target, test_data, train_size=0.7, random_state=SEED, stratify=test_data)
# x_train, x_val, y_train, y_val = train_test_split(train_target, train_data, train_size=0.8, random_state=SEED, stratify=train_data)

# lgbm_wrapper = LGBMClassifier(n_estimators=1000)

# # LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
# evals = [(x_test, y_test)]
# lgbm_wrapper.fit(x_train, y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=evals, verbose=True)
# preds = lgbm_wrapper.predict(x_test)
# pred_proba = lgbm_wrapper.predict_proba(x_test)[:, 1]

# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.metrics import precision_score, recall_score
# from sklearn.metrics import f1_score, roc_auc_score

# def get_clf_eval(y_test, pred=None, pred_proba=None):
#     confusion = confusion_matrix(y_test, pred)
#     accuracy = accuracy_score(y_test, pred)
#     precision = precision_score(y_test, pred)
#     recall = recall_score(y_test, pred, labels=None, pos_label=1, average='weighted')
#     f1 = f1_score(y_test, pred)
#     roc_auc = roc_auc_score(y_test, pred)
#     print('오차 행렬')
#     print(confusion)
#     print(f'정확도 : {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, F1 : {f1:.4f}, AUC:{roc_auc:4f}')

# get_clf_eval(y_test, preds, pred_proba)





# 
# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(lgbm_wrapper, ax=ax)
# x_train, x_temp, y_train, y_temp = train_test_split(train_target, train_data, train_size=0.4, random_state=SEED,stratify=train_data)
# x_test, x_temp, y_test, y_temp = train_test_split(test_target, test_data, train_size=0.4, random_state=SEED,stratify=test_data)

# x_train, x_val, y_train, y_val = train_test_split(train_target, train_data, train_size=0.8, random_state=SEED, stratify=train_data)

# train_list=random.sample(list(range(x_train.shape[0])), int(round(x_train.shape[0]*0.01,0)) )
# # x_train=train_target
# # x_train=x_train.iloc[train_list,:]
# # y_train=train_data
# # y_train=y_train.iloc[train_list,:]
# # x_test=train_target
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, r2_score
# # from sklearn.utils import all_estimators
# import warnings
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# # Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# warnings.filterwarnings('ignore')
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
# from sklearn.pipeline import Pipeline, make_pipeline
# #RandomForest : 상당히 중요하다!
# #머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
# #acc_score = 분류. r2 = 회귀
# from xgboost import XGBClassifier, XGBRegressor, plot_importance
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


# parameters_arr = [
#     {"anyway__n_estimators":[100,200,300], "anyway__learning_rate":[0.1,0.3,0.01,0.001], 
#     "anyway__max_depth":[4,5,6]},
#     {"anyway__n_estimators":[90,100,110], "anyway__learning_rate":[0.1,0.01,0.001], 
#     "anyway__max_depth":[4,5,6], "anyway__colsample_bytree":[0.6,0.9,1]},
#     {"anyway__n_estimators":[9,110], "anyway__learning_rate":[0.1,0.001,0.5], 
#     "anyway__max_depth":[4,5,6], "anyway__colsample_bytree":[0.6,0.9,1], 
#     "anyway__colsample_bylevel":[0.6,0.7,0.9]}
# ]
# n_jobs = -1

# kfold = KFold(n_splits=5, shuffle=True)
# pipe = Pipeline([('scaler', StandardScaler()),('anyway', LGBMClassifier() )])
# model = RandomizedSearchCV(pipe, parameters_arr, cv=kfold, verbose=0)
# model.fit(x_train, y_train)
# score_at_fi = model.score(x_test, y_test)
# print("original score:", score_at_fi)
# original_params_at_fi = model.best_params_
# print("original 최적의 파라미터:", original_params_at_fi)

# best_model = model.best_estimator_
# score_at_fi_param = best_model.score(x_test, y_test)
# print("find param score:", score_at_fi_param)

# y_predict = model.predict(x_test) #예측값 생성 후, test['18~20_ride']에 집어 넣는다.
# print("y_predict : ",y_predict)

#test[['id','18~20_ride']].to_csv("./mini_project/datadacon_base_middle2.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 내보낸다


# test[['id','18~20_ride']].to_csv("dacon_base_middle2.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 내보낸다












###################################################################################
# target_col = '18~20_ride'
# train_label = train[target_col]


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

# # Fold별로 학습진행
# for ind, (trn_ind, val_ind) in tqdm_notebook(enumerate( kfolds.split( X = train, y = train_label ) ) ):
    
#     # Train/Valid-set을 정의
#     x_train , y_train = train.iloc[trn_ind], train_label[trn_ind]
#     X_valid , y_valid = train.iloc[val_ind], train_label[val_ind]
    
#     # Light GBM
#     print("---TRAINING---")
    
#     # dtrain/dvalid 정의
#     dtrain = lgbm.Dataset(x_train, y_train)
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

# x_train=train
# y_train=train_label
# x_test=test

# print("x train shape : ", train.shape) #(1061878, 67)
# print("y train shape : ", test.shape) #(310121, 67)
# train = train.T
# test = test.T
# print(train.shape)
# print(test.shape)
# x_train_poly = train.fit_transform([[i] for i in x_train])
# x_train_poly.shape
# # (11, 4)
# x_train, x_test, y_train, y_test = train_test_split(test, train, test_size=0.2, random_state=156)
# target_col = train['18~20_ride']
# train_label = train[target_col]

# train = train.drop('18~20_ride', axis=1)

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
# from tqdm.notebook import tqdm
# # Fold별로 학습진행
# for ind, (trn_ind, val_ind) in tqdm(enumerate( kfolds.split( X = train, y = train_label ) ) ):
    
#     # Train/Valid-set을 정의
#     x_train , y_train = train.iloc[trn_ind], train_label[trn_ind]
#     X_valid , y_valid = train.iloc[val_ind], train_label[val_ind]
    
#     # Light GBM
#     print("---TRAINING---")
    
#     # dtrain/dvalid 정의
#     dtrain = lgbm.Dataset(x_train, y_train)
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

# x_train=train
# y_train=train_label
# x_test=test
# print(x_train,"x_train")
# print(y_train,"y_train")
# print(x_test,"x_test")
'''

pca1=PCA(n_components=12)
train=pca1.fit_transform(train)
test=pca1.fit_transform(test)
#y=train['18~20_ride']
#x=train.drop('18~20_ride', axis=1)
learning_rate=0.01
n_estimators=100
colsample_bylevel=1
colsample_bytree=1
print("x train shape : ", train.shape) #(1061878, 67)
print("y train shape : ", test.shape) #(310121, 67)
max_depth=list(range(2,10))
n_jobs=8
model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
                     n_estimators=n_estimators, n_jobs=n_jobs,
                     colsample_bylevel=colsample_bylevel,
                     colsample_bytree=colsample_bytree)

x_train, x_test, y_train, y_test = train_test_split(test, train,shuffle=True, train_size=0.8, random_state=num)

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,target, train_size=0.8, random_state=77)/

#random 20% 데이터
print("x train shape : ", x_train.shape) 
print("y train shape : ", y_train.shape)
print("x test shape : ", x_test.shape) 
print("y test shape : ", y_test.shape) 


parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.01,0.001], 
    "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[9,110], "learning_rate":[0.1,0.001,0.5], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], 
    "colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1
#2. 모델
kfold=KFold(n_splits=5, shuffle=True)
model=RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold)


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict=model.predict(x_test)
print('최종정답률 : ', r2_score(y_test, y_predict))
print('feature_importance : ', XGBRegressor().fit(x_train, y_train).feature_importances_)



x_train=train2
random.seed(1217) #동일한 샘플링하기 위한 시드번호
train_list=random.sample(list(range(x_train.shape[0])), int(round(x_train.shape[0]*0.01,0)) )
x_train=x_train.iloc[train_list,:]
y_train=train2[target]
y_train=y_train.iloc[train_list,:]

x_test=test

param_grid = {
     'max_features': [2,3,5],
     'min_samples_leaf': [2,3],
     'min_samples_split': [2,4,6],
     'n_estimators': [100, 200,500]
}

# rf = RandomForestRegressor(random_state=1217) # 랜덤포레스트 모델을 정의한다.
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid) # GridSearchCV를 정의한다.

# grid_search.fit(x_train, y_train)

# grid_search.best_params_ #학습 이후 최적의 paramter를 출력
# x_train=train2[input_var]
# y_train=train2[target]

# x_test=test2[input_var]
# rf = RandomForestRegressor(max_features=3,min_samples_leaf=2,min_samples_split=2,n_estimators=500,random_state=1217)

# rf.fit(x_train,y_train) #학습 

# test['18~20_ride'] = rf.predict(x_test) #예측값 생성 후, test['18~20_ride']에 집어 넣는다.

# y_predict=test['18~20_ride']

# #100번 돈다.
# print("최적의 매개변수",rf.best_estimator_) #모델 최고의 평가자

# print("최종정답률",accuracy_score(y_test,y_predict))

# # test[['id','18~20_ride']].to_csv("./mini_project/model/middle2.csv",index=False) # id와 18~20_ride만 선택 후 csv 파일로 내보낸다
'''