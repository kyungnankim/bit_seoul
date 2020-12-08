import pandas as pd
import numpy as np
df_aug = pd.read_csv('./mini_project/data/제주공항_8월_강수량.csv')
df_sep = pd.read_csv('./mini_project/data/제주공항_9월_강수량.csv')
df_oct = pd.read_csv('./mini_project/data/제주공항_10월_강수량.csv')

hour_cols = df_sep.columns[1:]

for col in hour_cols:
    df_aug[col] = df_aug[col].replace({0.0 : 0.1})
    df_sep[col] = df_sep[col].replace({0.0 : 0.1})
    df_oct[col] = df_oct[col].replace({0.0 : 0.1})

df_aug['daily_rain'] = df_aug[hour_cols].sum(axis=1)
df_sep['daily_rain'] = df_sep[hour_cols].sum(axis=1)
df_oct['daily_rain'] = df_oct[hour_cols].sum(axis=1)

df_aug['date'] = pd.date_range(start='2020-08-01', end='2020-08-31')
df_sep['date'] = pd.date_range(start='2020-09-01', end='2020-09-30')
df_oct['date'] = pd.date_range(start='2020-10-01', end='2020-10-31')

df_aug = df_aug[df_aug['date']=='2020-08-31']
df_oct = df_oct[df_oct['date']=='2020-10-16']

df = pd.concat([df_aug, df_sep],axis=0)
df = pd.concat([df, df_oct],axis=0)


# df=df.to_numpy()


# #데이터 스케일링
# from sklearn. preprocessing import StandardScaler, MinMaxScaler
# scaler1=StandardScaler()
# scaler1.fit(df)
# df=scaler1.transform(df)

# df_npy = df.drop(['df_aug','df_oct','df_aug'], axis=1).to_numpy()
# np.save('./mini_project/data/daily_rain.npy', arr=daily_rain)
df[['date','daily_rain']].to_csv('./mini_project/data/daily_rain.csv',index=False)
# df=df.astype('float32')
