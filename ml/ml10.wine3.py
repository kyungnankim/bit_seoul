#winequality-white.csv

import numpy as np
import pandas as pd

# 1. 데이터
#pandas로 csv 불러오기
wine = pd.read_csv('./data/csv/winequality-white.csv', header=0, index_col=None, sep=';')
count_data = wine.groupby('quality')['quality'].count()
print(count_data)