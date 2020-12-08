import pandas as pd

df_sep = pd.read_csv('./mini_project/data/제주공항_9월_전운량.csv')
df_oct = pd.read_csv('./mini_project/data/제주공항_10월_전운량.csv')

hour_cols = ['0H', '1H', '2H', '3H', '4H', '5H',
             '6H', '7H', '8H','9H', '10H', '11H']
df_sep = df_sep[['Unnamed: 0'] + hour_cols]
df_oct = df_oct[['Unnamed: 0'] + hour_cols]

for col in hour_cols:
    df_sep[col] = df_sep[col].replace({0.0 : 0.1})
    df_oct[col] = df_oct[col].replace({0.0 : 0.1})

# df_sep['hourly_rain'] = df_sep[hour_cols].sum(axis=1)
# df_oct['hourly_rain'] = df_oct[hour_cols].sum(axis=1)

# df_sep['date'] = pd.date_range(start='2020-09-01', end='2020-09-30')
# df_oct['date'] = pd.date_range(start='2020-10-01', end='2020-10-31')

# df_oct = df_oct[df_oct['date']=='2020-10-16']

# df = pd.concat([df_sep, df_oct],axis=0)

# df[['date','hourly_rain']].to_csv('./mini_project/data/hourly_rain.csv',index=False)


df_sep['hourly_cloud'] = df_sep[hour_cols].sum(axis=1)
df_oct['hourly_cloud'] = df_oct[hour_cols].sum(axis=1)

df_sep['date'] = pd.date_range(start='2020-09-01', end='2020-09-30')
df_oct['date'] = pd.date_range(start='2020-10-01', end='2020-10-31')

df_oct = df_oct[df_oct['date']=='2020-10-16']

df = pd.concat([df_sep, df_oct],axis=0)

df[['date','hourly_cloud']].to_csv('./mini_project/data/hourly_cloud.csv',index=False)