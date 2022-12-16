# ematrix
The multidimensional matrix calculate package,  you can handle three-dimensional data just like pd.DataFrame handle two-dimensional data
Usually for financial quantification timeseries data


多维矩阵计算库，它能像pd.DataFrame处理二维数据那样处理三维数据，常用于金融量化时序类数据

pip install ematrix

It's beautiful right? ^_^

dm = Matrix()
print('data_matrix:', dm)
res = dm[dm['code'] == '000001']
print('data_matrix_slice:\n', res)
------------------------------
data_matrix:
             ['close', 'high', 'low', 'open', 'ret']
100
000001  [0.80676805 0.30489957...0.83385824 0.56749199]
000002  [0.0158474  0.65408549...0.14293321 0.76696439]
                  ......

200
000001  [0.62020139 0.96969446...0.76984185 0.14326731]
000002  [0.82182821 0.10924971...0.9967727  0.51963606]
                  ......
                  ......
400
...
000003  [0.15914842 0.99641071...0.15914842 0.99641071]
Dimension: ['timestamp', 'code', 'features'], Shape: (4, 3, 5), dtype: matrix

data_matrix_slice:
         close      high       low      open       ret
100  0.806768  0.304900  0.210918  0.833858  0.567492
200  0.620201  0.969694  0.754948  0.769842  0.143267
300  0.008385  0.087556  0.994873  0.706558  0.752703
400  0.388965  0.946414  0.967339  0.567995  0.071485


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

