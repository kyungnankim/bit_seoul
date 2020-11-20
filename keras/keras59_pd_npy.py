#iris_ys2.csv 파일을 넘파일로 불러오기

#불러온 데이터를 판다스로 저장하시오,

import pandas as pd
import numpy as np
from pandas import DataFrame
csv_data = np.loadtxt('./data/csv/iris_ys2.csv', delimiter=',')
x=csv_data[:,:4]
y=csv_data[:,4]

df = pd.DataFrame(data=csv_data, index=None, columns=None)
print(type(df))
print(df.shape)

df.to_csv('./data/csv/iris_ys2_pd.csv', mode='a', header=False)

print(x.shape)
print(y.shape)