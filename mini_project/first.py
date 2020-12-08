# prevent overfit
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Library 예시
import os

# handling
import pandas as pd
import numpy as np
import datetime
import random
import gc
from tqdm import tqdm_notebook as tqdm

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# prevent overfit
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# model
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

SEED=42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def rmse(y_true, y_pred):
    return np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)

########################### BASIC SETTING

seed_everything(SEED)
TARGET = '18~20_ride'

########################### DATA LOAD
train_df = pd.read_csv('./mini_project/data/train.csv')
test_df = pd.read_csv('./mini_project/data/test.csv')
sub = pd.read_csv('./mini_project/data/submission_sample.csv')

def df_copy(tr_df, te_df):
    tr = tr_df.copy();te = te_df.copy()
    return tr, te

def base_preprocessing(tr_df, te_df):
    tr, te = df_copy(tr_df, te_df)

    tr['bus_route_id'] = tr['bus_route_id'].apply(lambda x: str(x)[:-4]).astype(int)
    te['bus_route_id'] = te['bus_route_id'].apply(lambda x: str(x)[:-4]).astype(int)
    tr['station_name2'] = tr['station_name'].apply(lambda x: str(x)[:2])
    te['station_name2'] = te['station_name'].apply(lambda x: str(x)[:2])
    tr['station_name'] = tr['station_name'].apply(lambda x: x.replace(' ', ''))
    te['station_name'] = te['station_name'].apply(lambda x: x.replace(' ', ''))

    le = LabelEncoder().fit(pd.concat([tr['station_name'], te['station_name']]))
    le2 = LabelEncoder().fit(pd.concat([tr['station_name2'], te['station_name2']]))
    for df in [tr, te]:
        df['day'] = pd.to_datetime(df['date']).dt.day
        df['week'] = pd.to_datetime(df['date']).dt.week
        df['weekday'] = pd.to_datetime(df['date']).dt.weekday
        df['station_name'] = le.transform(df['station_name'])
        df['station_name2'] = le2.transform(df['station_name2'])

        df['6~8_ride'] = df[['6~7_ride','7~8_ride']].sum(1)
        df['6~9_ride'] = df[['6~7_ride','7~8_ride','8~9_ride']].sum(1)
        df['6~10_ride'] = df[['6~7_ride','7~8_ride','8~9_ride', '9~10_ride']].sum(1)
        df['6~8_takeoff'] = df[['6~7_takeoff','7~8_takeoff']].sum(1)
        df['6~9_takeoff'] = df[['6~7_takeoff','7~8_takeoff','8~9_takeoff']].sum(1)
        df['6~10_takeoff'] = df[['6~7_takeoff','7~8_takeoff','8~9_takeoff', '9~10_takeoff']].sum(1)
    te['day'] = te['day']+30
    return tr, te

def lat_long_create(tr_df, te_df):
    tr, te = df_copy(tr_df, te_df)
    tr['lat_long'] = np.round(tr['latitude'], 2).astype(str) + np.round(tr['longitude'], 2).astype(str)
    te['lat_long'] = np.round(te['latitude'], 2).astype(str) + np.round(te['longitude'], 2).astype(str)
    le = LabelEncoder().fit(pd.concat([tr['lat_long'], te['lat_long']]))
    tr['station_lat_long'] = le.transform(tr['lat_long'])
    te['station_lat_long'] = le.transform(te['lat_long'])

    tr['lat_long'] = np.round(tr['latitude'], 3).astype(str) + np.round(tr['longitude'], 2).astype(str)
    te['lat_long'] = np.round(te['latitude'], 3).astype(str) + np.round(te['longitude'], 2).astype(str)
    le = LabelEncoder().fit(pd.concat([tr['lat_long'], te['lat_long']]))
    tr['station_lat_long2'] = le.transform(tr['lat_long'])
    te['station_lat_long2'] = le.transform(te['lat_long'])
    return tr, te

def feature_combine(tr_df, te_df):
    tr, te = df_copy(tr_df, te_df)
    for df in [tr, te]:
        df['bus_route_id_station_code'] = ((df['bus_route_id']).astype(str) + (df['station_code']).astype(str)).astype('category')
        df['bus_route_id_station_lat_long'] = ((df['bus_route_id']).astype(str) + (df['station_lat_long']).astype(str)).astype('category')
    return tr, te

def category_transform(tr_df, te_df, columns):
    tr, te = df_copy(tr_df, te_df)
    for df in [tr, te]:
        df[columns] = df[columns].astype(str).astype('category')
    return tr, te

def frequency_encoding(tr_df, te_df, columns, normalize=False):
    tr, te = df_copy(tr_df, te_df)
    for col in columns:
        if not normalize:
            freq_encode = pd.concat([tr[col], te[col]]).value_counts()
            tr[col+'_fq_enc'] = tr[col].map(freq_encode)
            te[col+'_fq_enc'] = te[col].map(freq_encode)
        else:
            freq_encode = pd.concat([tr[col], te[col]]).value_counts(normalize=True)
            tr[col+'_fq_enc_nor'] = tr[col].map(freq_encode)
            te[col+'_fq_enc_nor'] = te[col].map(freq_encode)
    return tr, te

def remove_outlier(tr_df, te_df, columns):
    tr, te = df_copy(tr_df, te_df)
    for col in columns:
        tr[col] = np.where(tr[col].isin(te[col]), tr[col], 0)
        te[col] = np.where(te[col].isin(tr[col]), te[col], 0)
    return tr, te

def day_agg(tr_df, te_df, merge_columns, columns, aggs=['mean']):
    tr, te = df_copy(tr_df, te_df)
    for merge_column in merge_columns:
        for col in columns:
            for agg in aggs:
                valid = pd.concat([tr[[merge_column, col]], te[[merge_column, col]]])
                new_cn = merge_column + '_' + agg + '_' + col
                if agg=='quantile':
                    valid = valid.groupby(merge_column)[col].quantile(0.8).reset_index().rename(columns={col:new_cn})
                else:
                    valid = valid.groupby(merge_column)[col].agg([agg]).reset_index().rename(columns={agg:new_cn})
                valid.index = valid[merge_column].tolist()
                valid = valid[new_cn].to_dict()
            
                tr[new_cn] = tr[merge_column].map(valid)
                te[new_cn] = te[merge_column].map(valid)
    return tr, te

def sub_day_agg(tr_df, te_df, merge_columns, date_columns, columns, aggs=['mean']):
    tr, te = df_copy(tr_df, te_df)
    for merge_column in merge_columns:
        for date in date_columns:
            tr['mc_date'] = tr[merge_column].astype(str) + '_' +tr[date].astype(str)
            te['mc_date'] = te[merge_column].astype(str) + '_' +te[date].astype(str)
            for col in columns:
                for agg in aggs:
                    valid = pd.concat([tr[['mc_date', col]], te[['mc_date', col]]])
                    new_cn = merge_column + '_' + date + '_' + col + '_' + agg
                    if agg=='quantile':
                        valid = valid.groupby('mc_date')[col].quantile(0.8).reset_index().rename(columns={col:new_cn})
                    else:
                        valid = valid.groupby('mc_date')[col].agg([agg]).reset_index().rename(columns={agg:new_cn})
                    valid.index = valid['mc_date'].tolist()
                    valid = valid[new_cn].to_dict()
                
                    tr[new_cn] = tr['mc_date'].map(valid)
                    te[new_cn] = te['mc_date'].map(valid)
    tr = tr.drop(columns=['mc_date'])
    te = te.drop(columns=['mc_date'])
    return tr, te

########################### Final features list
remove_features = ['id', 'date', 'in_out', TARGET]
ride_take = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff']

remove_features += ['day', 'week', 'weekday', 'lat_long']
tr, te = base_preprocessing(train_df, test_df)
tr, te = lat_long_create(tr, te)
tr, te = feature_combine(tr, te)

ride_take += ['6~8_ride', '6~9_ride', '6~10_ride', '6~8_takeoff', '6~9_takeoff', '6~10_takeoff']

tr, te = day_agg(tr, te, merge_columns=['day'], columns=ride_take, aggs=['mean'])
tr, te = sub_day_agg(tr, te, merge_columns=['bus_route_id', 'station_code', 'station_lat_long'], date_columns=['day'], columns=ride_take, aggs=['mean'])
tr, te = sub_day_agg(tr, te, merge_columns=['bus_route_id', 'station_code', 'station_name', 'station_lat_long'], date_columns=['day'], columns=ride_take, aggs=['quantile'])

category_features = ['bus_route_id', 'station_code', 'station_name', 'station_name2', 'station_lat_long', 'station_lat_long2', 'bus_route_id_station_code', 'bus_route_id_station_lat_long']
tr, te = frequency_encoding(tr, te, category_features)


print(tr.head())


########################### Model
def make_predictions(model, tr_df, tt_df, features_columns, target, params, category_feature=[''], NFOLDS=4, oof_save=False, clip=999, SEED=SEED):
    X,y = tr_df[features_columns], tr_df[target]
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(tr_df))
    pred = np.zeros(len(tt_df))
    fi_df = pd.DataFrame()
    
    for fold_, (trn_idx, val_idx) in enumerate(kf.split(X)):
        print('Fold:',fold_)
        tr_data = lgb.Dataset(X.loc[trn_idx], label=y[trn_idx].clip(0, clip))
        vl_data = lgb.Dataset(X.loc[val_idx], label=y[val_idx])
        if model=='lgb':
            estimator = lgb.train(params, tr_data, valid_sets = [tr_data, vl_data], verbose_eval = 500)
            fi_df = pd.concat([fi_df, pd.DataFrame(sorted(zip(estimator.feature_importance(), features_columns)), columns=['Value', 'Feature'])])
        
        oof[val_idx] = estimator.predict(X.loc[val_idx])
        pred += estimator.predict(tt_df[features_columns])/NFOLDS
        del estimator
        gc.collect()

    oof = np.where(oof&gt;0, oof, 0)
    pred = np.where(pred&gt;0, pred, 0)

    if oof_save:
        if model=='lgb':
            np.save('./mini_project/data/oof_lgb.npy', oof)
            np.save('./mini_project/data/pred_lgb.npy', pred)
        elif model=='cat':
            np.save('./mini_project/data/oof_cat.npy', oof)
            np.save('./mini_project/data/pred_cat.npy', pred)

    tt_df[target] = pred
    print('OOF RMSE:', rmse(y, oof))
    
    try:
        fi_df = fi_df.groupby('Feature').mean().reset_index().sort_values('Value')
    except:
        pass

    return tt_df[['id', target]], fi_df
## -------------------

lgb_params = {
        'objective':'regression',
        'boosting_type':'gbdt',
        'metric':'rmse',
        'n_jobs':-1,
        'learning_rate':0.003,
        'num_leaves': 700,
        'max_depth':-1,
        'min_child_weight':5,
        'colsample_bytree': 0.3,
        'subsample':0.7,
        'n_estimators':50000,
        'gamma':0,
        'reg_lambda':0.05,
        'reg_alpha':0.05,
        'verbose':-1,
        'seed': SEED,
        'early_stopping_rounds':50
    }
tr, te = remove_outlier(tr, te, category_features)
tr, te = category_transform(tr, te, category_features)

features_columns = [col for col in tr.columns if col not in remove_features]

test_predictions, fi = make_predictions('lgb', tr, te, features_columns, TARGET, lgb_params, category_feature=category_features, NFOLDS=5, oof_save=True)

test_predictions.to_csv('lgb_model.csv', index=False)