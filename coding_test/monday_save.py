import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/samsung_1120.csv', header=0, index_col=0, encoding='CP949', sep=',' )
hite = pd.read_csv('./data/csv/bit_1120.csv', header=0, index_col=0, encoding='CP949',sep=',' )
gold = pd.read_csv('./data/csv/금현물.csv', header=0, index_col=0, encoding='CP949', sep=',' )
kosdaq = pd.read_csv('./data/csv/kosdaq.csv', header=0, index_col=0, encoding='CP949',sep=',' )

samsung = samsung[['시가','종가','저가','고가', '거래량', '기관', '외인(수량)']]
samsung = samsung.iloc[0:626]

hite = hite[['시가','종가','저가','고가', '금액(백만)', '기관']]
hite = hite.iloc[0:626]

gold = gold[['시가','종가','저가','고가', '거래대금(백만)']]
gold = gold.iloc[0:625]

kosdaq = kosdaq[['시가','저가','고가']]
kosdaq = kosdaq.iloc[0:626]

samsung = samsung.sort_values(['일자'],ascending=['True'])
hite = hite.sort_values(['일자'],ascending=['True'])
gold = gold.sort_values(['일자'],ascending=['True'])
kosdaq = kosdaq.sort_values(['일자'],ascending=['True'])

# 콤마 제거 후 문자를 정수로 변환
def  split_x(data):
    for i in range(len(data.index)):
        for j in range(len(data.iloc[i])):
            data.iloc[i,j]=int(data.iloc[i,j].replace(',',''))
            # print("i,j:",i,"/",j)
    return data
samsung = split_x(samsung)
hite = split_x(hite)
gold = split_x(gold)
kosdaq = split_x(kosdaq)

samsung_target = samsung['시가']
samsung_target = samsung_target.to_numpy()
np.save('./data/samsung_target.npy',arr=samsung_target)
samsung_data = samsung.to_numpy()
np.save('./data/samsung_data.npy',arr=samsung_data)

hite_target = hite['시가']
hite_target = hite_target.to_numpy()
np.save('./data/hite_target.npy',arr=hite_target)
hite_data = hite.to_numpy()
np.save('./data/hite_data.npy',arr=hite_data)

gold_target = gold['시가']
gold_target = gold_target.to_numpy()
np.save('./data/gold_target.npy',arr=gold_target)
gold_data = gold.to_numpy()
np.save('./data/gold_data.npy',arr=gold_data)

kosdaq_target = kosdaq['시가']
kosdaq_target = kosdaq_target.to_numpy()
np.save('./data/hite_target.npy',arr=kosdaq_target)
kosdaq_data = kosdaq.to_numpy()
np.save('./data/kosdaq_data.npy',arr=kosdaq_data)