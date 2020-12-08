import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.utils import all_estimators
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# Classifier :  분류, Regressor : 회귀모델 단 logit Regression 는 분류
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
#RandomForest : 상당히 중요하다!
#머신러닝 : 데이터 → 모델 → 컴파일 → 평가예측
#acc_score = 분류. r2 = 회귀
from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance

#load data
# x_train = pd.read_csv('./mini_project/data/hourly_rain.csv', header=0, index_col=None,sep=',')
# x_train = pd.read_csv('./mini_project/data/제주공항_9월_강수량.csv')
# y_train = pd.read_csv('./mini_project/data/제주공항_10월_강수량.csv')
x_train = pd.read_csv('./mini_project/data/hourly_rain.csv')

def grap_year(data):
    data = str(data)
    return int(data[:5])

def grap_month(data):
    data = str(data)
    return int(data[6:8])

def grap_date(data):
    data = str(data)
    return int(data[9:])

# 날짜 처리
data = x_train.copy()
# data = data.fillna('')
print(data.head())

data['year'] = data.apply(lambda x: grap_year(x_train))
data['month'] = data.apply(lambda x: grap_month(x_train))
data['date'] = data.apply(lambda x: grap_date(x_train))
# data = data.drop(, axis=1)
data.head()


x=x_train.drop('date', axis=1)
y=x_train['date']

print(x.shape) #(4898, 11)
print(y.shape) #(4898,)
'''
x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)

kfold = KFold(n_splits=5, shuffle=True)
params = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.01,0.001], 
    "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[9,110], "learning_rate":[0.1,0.001,0.5], 
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], 
    "colsample_bylevel":[0.6,0.7,0.9]}
]
n_jobs = -1

model = RandomizedSearchCV(XGBRegressor(), params, n_jobs=n_jobs, cv=kfold, verbose=2)

model.fit(x_train,y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적 하이퍼 파라미터 : ", model.best_params_)
print("최고 정확도 : {0:.4f}".format(model.best_score_))

r2 = model.score(x_test,y_test)
print("r2 : ", r2)


'''

#파이썬에서 피클을 사용해 객체 배열(numpy 배열)을 저장할 수 있음 -> 배열의 내용이 일반 숫자 유형이 아닌 경우 (int/float) pickle를 사용해 array 저장
#shape
# print("x train shape : ", x_train.shape) #(590540, 367)
# print("y train shape : ", y_train.shape) #(590540,)
# # print("x test shape : ", test.shape) #(506691, 367)
# x_train =x_train.reshape(x_train.shape[1:])
# y_train =y_train.reshape(y_train.shape[1:])
# #모델 테스트를 위해 부분 데이터 잘라서 사용 (데이터 양이 너무 많음) - random으로 20%
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.7, shuffle=True)

# # test, test_temp = train_test_split(test, train_size=0.01, random_state=77)

# #random 20% 데이터
# print("x train shape : ", x_train.shape) #(118108, 367)
# print("y train shape : ", y_train.shape) #(118108,)
# print("x test shape : ", x_test.shape) #(101338, 367)
# print("y test shape : ", y_test.shape) #(101338, 367)


# # params = {
# #     "n_estimators":[500, 800, 1000, 1200], # n_estimators default = 100 (learning rate를 낮게 잡아줬으니까 충분한 학습을 위해 늘려줌)
# #     "learning_rate":[0.01, 0.05, 0.001], # learning_rate default = 0.1
# #     "max_depth":range(3,10,3), # max_depth default = 6
# #     "colsample_bytree":[0.5,0.6,0.7], # colsample_bytree default = 1 (항상 모든 나무에서 중요한 칼럼에만 몰두해서 학습 -> 과적합 위험) / 학습할 칼럼 수가 많기 때문에 0.5-0.7까지 잡음    
# #     "colsample_bylevel":[0.6,0.7,0.9],
# #     'min_child_weight':range(1,6,2),
# #     'subsample' :  [0.6, 0.8] 
# #     # 'objective' : ['binary:logistic'],
# #     # 'eval_metric' : ['auc'],
# #     # 'tree_method' : ['gpu_hist']
# #     }



# # # model.fit(x, y)


# # # from sklearn.model_selection import KFold
# # # from sklearn.model_selection import cross_val_score
# # # k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# # # scoring = 'accuracy'
# # # score = cross_val_score(model, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
# # # print(score)

# # # round(np.mean(score)*100, 2)

# # # predict = model.predict([one,ten])
# # # result =pd.DataFrame({'date': ten['date'], '8시승차총승객수': predict})
# # # result.to_csv('./mini_project/data/result.csv', index=False) 


# # kfold = KFold(n_splits=5, shuffle=True)
# # skfold = StratifiedKFold(n_splits=5, shuffle=True)
# # xgb = xgboost.XGBClassifier(tree_method='gpu_hist', 
# #                             predictor='gpu_predictor',
# #                             objective= 'binary:logistic',
# #                             eval_metric= 'auc'
# #                             )

# # model = RandomizedSearchCV(xgb, params, cv=skfold, verbose=1, scoring=scoring, n_iter=10, refit='AUC', return_train_score=True, random_state=77)
# # # Scoring: 평가 기준으로 할 함수 / cv: int, 교차검증 생성자 또는 반복자 / n_iter: int, 몇 번 반복하여 수행할 것인지에 대한 값
# # # model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=n_jobs, cv=5, verbose=1, scoring=scoring, refit="AUC")

# # model.fit(x_train,y_train)

# # df = pd.DataFrame(model.cv_results_)
# # # print("cv result : \n", df.loc[:,['mean_test_score', 'params']])

# # print("최적 하이퍼 파라미터 : ", model.best_params_)
# # print("최고 AUC : {0:.4f}".format(model.best_score_))

# # model = model.best_estimator_

# # result = model.predict(x_test)
# # sc = model.score(x_test,y_test)
# # print("score : ", sc)

# # acc = accuracy_score(y_test,result)
# # print("acc : ", acc)

# # result2 = model.predict_proba(x_test)[:,1]
# # roc = roc_auc_score(y_test, result2)
# # print("AUC : %.4f%%"%(roc*100))
# # #roc curve 그리기
# # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, result2)
# # roc_auc = auc(false_positive_rate, true_positive_rate)
# # print("roc_auc :", roc_auc)

# # plt.figure(figsize=(10,10))
# # plt.title('Receiver Operating Characteristic')
# # plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
# # plt.legend(loc = 'lower right')
# # plt.plot([0, 1], [0, 1],linestyle='--')
# # plt.axis('tight')
# # plt.ylabel('True Positive Rate')
# # plt.xlabel('False Positive Rate')
# # plt.show()

# # print(model.feature_importances_)
# # print(np.sort(model.feature_importances_)[:10])
# # print(np.sort(model.feature_importances_))
# # print(index[:10])

# # plot_importance(model, max_num_features=20)
# # plt.show()

# # #Available importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
# # # fig, ax = plt.subplots(3,1,figsize=(14,30))
# # # nfeats = 15
# # # importance_types = ['weight', 'cover', 'gain']

# # # for i, imp_i in enumerate(importance_types):
# # #     plot_importance(model, ax=ax[i], max_num_features=nfeats
# # #                     , importance_type=imp_i
# # #                     , xlabel=imp_i)
# # #     plt.show()

# # def plot_feature_importances(model):
# #     # n_features = x_train.shape[1]
# #     n_features = 10
# #     plt.figure(figsize=(10,10))
# #     plt.title('Model Feature Importances')
# #     feature_names = index
# #     sorted_idx = model.feature_importances_.argsort()[::-1]
# #     print("sorted_idx: ",sorted_idx)
# #     plt.barh(feature_names[sorted_idx][:20], model.feature_importances_[sorted_idx][:20], align='center')
# #     plt.xlabel("Feature Imortances", size=15)
# #     plt.ylabel("Feautres", size=15)
# #     plt.ylim(-1, n_features)

# # plot_feature_importances(model)
# # plt.show()


# # thresholds = np.sort(model.feature_importances_)
# # # print(thresholds)

# # save_score = 0
# # best_thresh = 0
# # for thresh in thresholds:
# #     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
# #     select_x_train = selection.transform(x_train)
    
# #     selection_model =  XGBClassifier(n_jobs=-1)
# #     selection_model.fit(select_x_train,y_train)
    
# #     select_test = selection.transform(test)
# #     y_predict = selection_model.predict(select_test)

# #     score =  model.score(test,y_predict)

# #     # print("Thresh=%.4f, n=%d, acc: %.4f%%" %(thresh, select_x_train.shape[1], score))

# #     if score > save_score:
# #         save_score = score
# #         best_thresh = thresh
# #     # print("best_thresh, save_score: ", best_thresh, save_score)

# # print("best_thresh, save_score: ", best_thresh, save_score)

# # selection = SelectFromModel(model, threshold=best_thresh, prefit=True)
# # x_train = selection.transform(x_train)
# # test = selection.transform(test)

# # model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=-1, cv=5)

# # model.fit(x_train,y_train)

# # print("최적 하이퍼 파라미터 : ", model.best_params_)
# # print("최고 정확도 : {0:.4f}".format(model.best_score_))

# # model = model.best_estimator_

# # result = model.predict(test)
# # acc = model.score(test,result)
# # print("acc : ", acc)