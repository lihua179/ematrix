# -*- coding:utf-8 -*-
# @文件名  :ematrix_demo.py
# @时间    :2022/12/4 22:34
# @作者    :Zed
# @目的

import pandas as pd
import os

"""
next plan:
        3. Increase logic operation such as '>= <= > < !=/~ and & or / in' : dm[dm['datetime']>=20220105 and dm['code'] in ['000001','000002']]       √
        4. Increase more __magic_function__ such as __del__: del dm["feature=='close'"] / dm.delete('datetime<=20220105')       √
        5. Del dm[dm[::3]:ClassMatrix]      √
        6. Increase function fill_na: dm.fill_na('forward_value/backward_value/zero_value/avg_value')       √
        7. Query function: dm.query("20220105<=datetime<=20220506 and code in ['000001','000002'] and feature.capital>float(1e9)")  doing...
        -1. More simular pd.DataFrame operation function in Matrix
            * dm.sort(dm['feature']==close,ascend=True) / dm.sort('datetime')
            * dm.resort_index('datetime') / dm.resort_value('datetime.20220510')
                _datetime='datetime.20220510'.split('.')[1]
                it will check dtype of dm.axis_dtype of datetime : data=float(_datetime) if self.axis_dict['datetime'].dtype==float else str(_datetime)
"""
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


"""
res:
------------------------------
res1

             ['ts_code', 'trade_date'...'ts_code', 'trade_date']
000001_SH
19901219  ['000001.SH' 19901219...1260.0 494.311]
19901220  ['000001.SH' 19901220...197.0 84.992]
                  ......

000009_SH
19901219  [nan nan...nan nan]
19901220  [nan nan...nan nan]
                  ......
                  ......
000045_SH
...
20221202  ['000045.SH' 20221202...'000045.SH' 20221202]
Index: code, Row: datetime, Columns: features, Shape: (3, 7808, 11), dtype: matrix

------------------------------
res2

             ['ts_code', 'trade_date'...'ts_code', 'trade_date']
19901219
000001_SH  ['000001.SH' 19901219...1260.0 494.311]
000009_SH  [nan nan...nan nan]
                  ......

19901220
000001_SH  ['000001.SH' 19901220...197.0 84.992]
000009_SH  [nan nan...nan nan]
                  ......
                  ......
20221202
...
000045_SH  ['000045.SH' 20221202...'000045.SH' 20221202]
Index: datetime, Row: code, Columns: features, Shape: (7808, 3, 11), dtype: matrix

------------------------------
res3
            ts_code trade_date      close  ... pct_chg         vol         amount
19901219        NaN        NaN        NaN  ...     NaN         NaN            NaN
19901220        NaN        NaN        NaN  ...     NaN         NaN            NaN
19901221        NaN        NaN        NaN  ...     NaN         NaN            NaN
19901224        NaN        NaN        NaN  ...     NaN         NaN            NaN
19901225        NaN        NaN        NaN  ...     NaN         NaN            NaN
...             ...        ...        ...  ...     ...         ...            ...
20221128  000045.SH   20221128  4776.5296  ... -0.4633  75751991.0   72706927.098
20221129  000045.SH   20221129  4846.1718  ...   1.458  96080088.0   92456933.081
20221130  000045.SH   20221130  4848.9355  ...   0.057  95022109.0   91497170.307
20221201  000045.SH   20221201   4873.663  ...    0.51  91828167.0  102087677.126
20221202  000045.SH   20221202  4851.5407  ... -0.4539  72540549.0   80947104.051

[7808 rows x 11 columns]
------------------------------
res4
             ts_code trade_date     close  ... pct_chg        vol       amount
000001_SH  000001.SH   20021226  1384.152  ... -2.6284  8652099.0  7055430.849
000009_SH        NaN        NaN       NaN  ...     NaN        NaN          NaN
000045_SH        NaN        NaN       NaN  ...     NaN        NaN          NaN

[3 rows x 11 columns]
------------------------------
res5
['000001_SH', '000009_SH', '000045_SH']
cost time: 0.09867095947265625


"""
