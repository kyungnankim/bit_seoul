from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
# import mglearn
import os
import matplotlib.pyplot as plt
import numpy as np
ram_prices = pd.read_csv('./mini_project/model/train_dist_name.csv',encoding='CP949', header=0,index_col=None,sep=',')

columns = ram_prices.columns
print(columns)
#  Index(['id', 'date', 'bus_route_id', 'in_out', 'station_code', 'station_name',    
#        'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
#        '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',     
#        '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff',
#        '18~20_ride', '68a', '810a', '1012a', 'location', 'dist_jeju',
#        'dist_gosan', 'dist_seongsan', 'dist_po', 'dust_name', 'temp',
#        'precipitation'],
#       dtype='object')
# # ram_prices = ram_prices.astype(int)
# data_train = ram_prices['precipitation']
columns_unique = [ '68a', '810a', '1012a']
# columns_unique = ['precipitation','bus_route_id','in_out','station_code',
#                   'station_name','latitude','longitude','6~7_ride','7~8_ride',
#                   '8~9_takeoff', '9~10_takeoff', '10~11_takeoff', '11~12_takeoff',
#                  'location', 'dist_jeju','dist_gosan', 'dist_seongsan',
#                  'dist_po', 'dust_name', 'temp' ]

unique_col = ram_prices[columns_unique]
ram_prices.drop(unique_col,axis=1,inplace=True) 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,8))
sns.countplot(x=ram_prices['18~20_ride']) 
plt.show()
