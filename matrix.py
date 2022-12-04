# -*- coding:utf-8 -*-
# @文件名  :ematrix_test.py
# @时间    :2022/12/4 22:34
# @作者    :Zed
# @目的

import pandas as pd
import os



def get_df_dict():
    data_list = os.listdir()
    data_list = [data.split('.')[0] for data in data_list if data.endswith('.csv')]

    _df_dict = {}
    for data in data_list:
        df = pd.read_csv(data + '.csv')
        df.index = df["trade_date"]  # 请确保dataframe.index是数据的时间,或者是有意义且不重复的值
        _df_dict[data] = df

    return _df_dict


df_dict = get_df_dict()

from ematrix import Matrix
import time

t1 = time.time()
data = Matrix(df_dict, axis_name=['code', 'datetime', 'features'])

print('-' * 30)
print('res1')
print(data)
data.T(['datetime', 'code', 'features'])
print('-' * 30)
print('res2')
print(data)

data0 = data[data['code'] == '000045_SH']
print('-' * 30)
print('res3')
print(data0)

data1 = data[data['datetime'] == 20021226]
print('-' * 30)
print('res4')
print(data1)

data2 = data[data['features'] == 'close']
print('-' * 30)
print('res5')
print(data.row)
t2 = time.time()
print('cost time:', t2 - t1)
