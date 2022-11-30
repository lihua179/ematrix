# -*- coding:utf-8 -*-
"""
像pandas DataFrame处理二维表格数据一样
Matrix类处理三维时间序列数据
目前没有赋值操作，因为时间序列大多用于读取，而不是修改
但是有拼接功能，新增的数据是dataframe类型可用matrix.update()或者matrix.append() 如果是matrix类型用matrix3=concat(matrix1,matrix2)
后续会更新更多功能，用于满足条件查询，定位，切片，矩阵批计算，因子批计算，使用体验同pandas.DataFrame
"""

import numpy as np
from typing import Union, NewType
import pandas as pd

set_print_limit = False  # 显示有限的输出设置
_SeriesType = Union[dict, tuple, list]

_SeriesTypeBase = [tuple, list]


def check_Series_unit(series):
    _len = 0
    _type = None
    _type0 = None
    _type1 = None
    if type(series) == list:
        for i, value in enumerate(series):

            if (type(value) == tuple or type(value) == list) and len(value) == 2:
                if not _len:
                    _len = len(value)
                elif _len != len(value):
                    raise ValueError(f'error length {series}')
                if not _type0:
                    # index类型
                    _type0 = type(value[0])
                    # value类型
                    _type1 = type(value[1])

                else:
                    if _type0 != type(value[0]) or _type1 != type(value[1]):
                        raise ValueError(f'error type {series} of {value}')
            else:
                if not _type:
                    _type = type(value)
                else:
                    if _type != type(value):
                        raise ValueError(f'error type {series}')

    if _type0:
        return 2, _type1
    else:
        return 1, _type


def check_Series_dtype(dtype):
    if dtype == str:
        dtype = 'str'
    elif dtype == float:
        dtype = 'float'
    elif dtype == int:
        dtype = 'int'
    elif dtype == bool:
        dtype = 'bool'
    elif dtype == dict:
        dtype = 'dict'
    return dtype


# axis_name = ['timestamp', 'code', 'features']
def _print(data: _SeriesType, head=0, tail=0):
    _print_len = 10
    if not head:
        __head = 5
    else:
        __head = head
    if not tail:
        __tail = 5
    else:
        __tail = tail
    _print_data = ''
    if type(data) == dict:
        _length = len(data.keys())
        if set_print_limit and _length > _print_len:
            keys = list(data.keys())
            up5 = keys[:__head]
            down5 = keys[-__tail:]
            for i, value in enumerate(up5):
                _print_data += f'{value} {data[value]}\n'
            if head:
                return _print_data
            _print_data += '...\n'
            _print_data_tail = ''
            for i, value in enumerate(down5):
                _print_data_tail += f'{value} {data[value]}\n'
            if tail:
                return _print_data_tail
            else:
                _print_data += _print_data_tail

        else:
            for key, value in data.items():
                _print_data += f'{key} {value}\n'

    elif type(data) == tuple or type(data) == list:
        _length = len(data)
        if set_print_limit and _length > _print_len:
            for i in range(__head):
                _print_data += f'{data[i][0]} {data[i][1]}\n'
            _print_data += '...\n'
            _print_data_tail = ''
            for i in range(__tail - 1, -1, -1):
                _print_data_tail += f'{data[i][0]} {data[i][1]}\n'
            if tail:
                return _print_data_tail
            else:
                _print_data += _print_data_tail
        else:

            for i in range(len(data)):
                _print_data += f'{data[i][0]} {data[i][1]}\n'

    else:
        raise ValueError(f'error type {data}')

    return _print_data


_loc = NewType('_loc', dict)
_SeriesDtype = Union[dict, tuple, str, int, float, list, _loc]


class Series:
    def __init__(self, series: list, name: str = 'Undefined', unique=False, astype: str = None):

        self.dtype = object
        self.astype = astype

        self.name = name
        self._series = series
        self._series_value = []
        self.length = len(self._series)
        self.index = list(range(self.length))
        self.unique = unique
        self._set_series()

    def _set_series(self):
        _res, _type = check_Series_unit(self._series)
        if _res == 1:
            self._series_value = np.array(self._series)
            self._series = list(zip(self.index, self._series_value))

        else:
            _series_values = []
            _series_values_index = []
            for i, value in enumerate(self._series):
                _series_values_index.append(value[0])
                _series_values.append(value[1])
            self.index = _series_values_index
            _series_values = np.array(_series_values)
            self._series = list(zip(self.index, _series_values))
            self._series_value = _series_values

        if self.astype:
            self.dtype = self.astype
        else:
            self.dtype = check_Series_dtype(_type)

    @property
    def values(self):
        return self._series_value

    def __eq__(self, other):

        _res = []
        if self.unique:
            for i, value in enumerate(self._series):

                if value[1] == other:
                    _res = [(self.name, i)]
                    return Series(_res, 'loc.unique', True, 'loc')
            if not _res:
                _res = [(self.name, None)]
                return Series(_res, 'loc.unique', True, 'loc')
        else:

            for value in self._series:
                if value[1] == other:
                    _res.append(True)
                else:
                    _res.append(False)

            return Series(_res, self.name)

    def __str__(self):
        _data = _print(self._series)
        info = f'Name: {self.name}, Length:{self.length}, dtype:{self.dtype}'
        return _data + info

    def __repr__(self):
        return self


# Series Demo————————————————————————————
data0 = ['a', 'b', 'c']
data = [(123, 'a'), (234, 'b'), (345, 'c'), (123, 'a'), (234, 'b'), (345, 'c'), (123, 'a'), (234, 'b'), (345, 'c'),
        (123, 'a'), (234, 'b'), (345, 'c')]
data = Series(data, 'code')

_MatrixGetSingleItemType = Union[str, int, Series]
_MatrixGetMultiItemType = Union[str, int, Series]
MatrixGetDataType = Union[np.ndarray, list, tuple, dict, None]


class DataFrame:
    def __init__(self, matrix_np: np.ndarray = None, axis_name: dict = None):
        # matrix_np:二维np矩阵， axis_name：行列名

        if not axis_name or matrix_np is None:
            # test data:
            axis_name = {'index': ['000001', '000002', '000003'],
                         'columns': ['close', 'high', 'low', 'open', 'ret']}
            matrix_np = np.random.random([3, 5])
        self.data = matrix_np
        self.axis_name = axis_name
        self.index = self.axis_name['index']
        self.columns = self.axis_name['columns']

    def to_pandas_df(self) -> pd.DataFrame:
        _df = pd.DataFrame(self.data)
        _df.index = self.index
        _df.columns = self.columns
        return _df

    def _print(self):
        pass

    def __str__(self):
        _df = self.to_pandas_df()
        return str(_df)

    def __repr__(self):
        _df = self.to_pandas_df()
        return _df


class Matrix:
    # 可接收多个pd.DataFrame聚合为xquant.Matrix数据类型
    # 标准样例：第一维度为时间，第二维度为合约标的代码，第三维度为特征因子features
    # 均需要外部提供，标准协议为：
    # axis_name={'timestamp':[xx,xx,xx],
    #            'code':[xx,xx,xx],
    #            'features':[xx,xx,xx],
    #           }
    # 需要解决一个问题：如果两个dataframe的长度不一致时如何解决
    def __init__(self, data: MatrixGetDataType = None, axis_name: dict = None, unique=True):
        # unique如果能确保值是唯一的，那么可提高性能
        self._dimension_dict = {}
        self.unique = unique

        if type(data) == np.ndarray:
            data: np.ndarray
            if len(data.shape) < 3:
                raise ValueError(f'error data dimension:{len(data.shape)}')
        # ToDo elif type(data) == np.ndarray:
        # list, tuple, dict 内部装着pd.dataframe

        if not axis_name or not data:
            # test data:
            axis_name = {'timestamp': [100, 200, 300, 400], 'code': ['000001', '000002', '000003'],
                         'features': ['close', 'high', 'low', 'open', 'ret']}
            data = np.random.random([4, 3, 5])
            # big data
            # axis_name = {'timestamp': list(range(1000)), 'code': list(range(5000)),
            #              'features': list(range(100))}
            # data = np.random.random([1000, 5000, 100])

        self.data = data
        self.axis_name = axis_name
        i = 0
        for key, value in axis_name.items():
            self._dimension_dict[key] = i
            i += 1
        self._axis_name_dict = {}
        self._axis_name_key = []
        self.init()

    def pandas_df_to_matrix(self, df_dict: dict):
        # 把多个pandas.dataframe数据放入df_dict中，然后循环处理为numpy.ndarray矩阵，最终变为matrix类

        pass

    def update(self):
        # Matrix加入新的DataFrame数据:
        # 为什么叫update?因为加入新的数据至少确保一个维度是相等的，另一个维度不相等但是可以在Matrix填充，
        # 这就需要对原有的所有Matrix.value都进行更新，因此叫更新，并且取这个名字让用户能明确自知加入的数据会让原Matrix数据维度可能发生变化
        pass

    def append(self):
        # 如果用户能确保加入的维度和Matrix.value维度一样，那么就不需要作维度相等判断，因此性能会更高
        pass

    def concat(self):
        # 拼接两个matrix数据
        pass

    def init(self):
        if len(self.data.shape) != 3:
            raise ValueError(f'error data shape {self.data}')
        if len(self.axis_name.keys()) != 3:
            raise ValueError(f'error axis_name len {self.data}')
        self.axis_name_to_range()

    def axis_name_to_range(self):
        _data_dict = {}
        for key, value in self.axis_name.items():
            # 用于实现类似loc精确定位矩阵元素
            _data_dict[key] = dict(zip(value, list(range(len(value)))))

        self._axis_name_dict = _data_dict
        self._axis_name_key = list(self._axis_name_dict.keys())

    # ToDo 像df.loc一样定位数据
    @property
    def loc(self):
        return

    # ToDo 像df.query一样条件查询数据
    def query(self):
        # res=matrix.query('100<=timestamp<=200 and close>21.5 or high>=100 and code=="000001.SZ"')
        #
        pass

    def __getitem__(self, *item):

        if len(item) == 1:
            item = item[0]

            item: _MatrixGetSingleItemType
            if type(item) == str:
                # dm['data_xx']:返回某维度所有index
                if item not in self._axis_name_key:
                    raise ValueError(f'{item} not exist in Matrix')

                return Series(list(self._axis_name_dict[item].keys()), item, self.unique)
            elif type(item) == Series:
                # Series类用于提供定位信息，根据dtype又分为bool和loc
                if item.dtype == 'bool':

                    if item.name not in self._dimension_dict:
                        raise ValueError(f'error {item.name} not in {self._dimension_dict}')
                    _dimension = self._dimension_dict[item.name]

                    if _dimension == 0:
                        _res = []

                        for i in range(len(item.values)):
                            if item.values[i]:
                                _res.append(self.data[i, :, :])
                        _res = np.array(_res)
                        _row_name = self._axis_name_key[1]
                        _columns_name = self._axis_name_key[2]
                        df_axis_name = dict(
                            index=self.axis_name[_row_name],
                            columns=self.axis_name[_columns_name]
                        )

                        return DataFrame(_res[0], df_axis_name)

                    elif _dimension == 1:
                        _res = []

                        for i in range(len(item.values)):
                            if item.values[i]:
                                _res.append(self.data[:, i, :])
                        _res = np.array(_res)
                        _row_name = self._axis_name_key[0]
                        _columns_name = self._axis_name_key[2]
                        df_axis_name = dict(
                            index=self.axis_name[_row_name],
                            columns=self.axis_name[_columns_name]
                        )

                        return DataFrame(_res[0], df_axis_name)
                    else:
                        _res = []

                        for i in range(len(item.values)):
                            if item.values[i]:
                                _res.append(self.data[:, :, i])
                        _res = np.array(_res)

                        _row_name = self._axis_name_key[0]
                        _columns_name = self._axis_name_key[1]
                        df_axis_name = dict(
                            index=self.axis_name[_row_name],
                            columns=self.axis_name[_columns_name]
                        )

                        return DataFrame(_res[0], df_axis_name)
                elif item.dtype == 'loc':

                    if not len(item.values):
                        return
                    if len(item.values) == 1:

                        _dimension = self._dimension_dict[item.index[0]]
                        if _dimension == 0:
                            _res = [self.data[item.values[0], :, :]]

                            _res = np.array(_res)
                            _row_name = self._axis_name_key[1]
                            _columns_name = self._axis_name_key[2]
                            df_axis_name = dict(
                                index=self.axis_name[_row_name],
                                columns=self.axis_name[_columns_name]
                            )

                            return DataFrame(_res[0], df_axis_name)

                        elif _dimension == 1:
                            _res = [self.data[:, item.values[0], :]]

                            _res = np.array(_res)
                            _row_name = self._axis_name_key[0]
                            _columns_name = self._axis_name_key[2]
                            df_axis_name = dict(
                                index=self.axis_name[_row_name],
                                columns=self.axis_name[_columns_name]
                            )

                            return DataFrame(_res[0], df_axis_name)

                        else:

                            _res = [self.data[:, :, item.values[0]]]

                            _res = np.array(_res)
                            _row_name = self._axis_name_key[0]
                            _columns_name = self._axis_name_key[1]
                            df_axis_name = dict(
                                index=self.axis_name[_row_name],
                                columns=self.axis_name[_columns_name]
                            )

                            return DataFrame(_res[0], df_axis_name)


            elif type(item) == list:

                if self.unique:
                    return

        else:

            print('_MatrixGetMultiItemType')
            pass

    @property
    def values(self):
        return self.data

        # ToDo
        # def __str__(self):
        #   return 像Series一样要对输出作限制，并且像多个DataFrame一样排序输出

        # dm=xquant.Matrix()  == pd.DataFrame()

    def __str__(self):
        return str(self.values)


def axis_name_to_range(data_dict):
    _data_dict = {}
    for key, value in data_dict.items():
        _data_dict[key] = dict(zip(value, list(range(len(value)))))

    return _data_dict


if __name__ == '__main__':
    # Matrix Demo————————————————————————————
    print('-' * 30)
    print('res')
    import time

    dm = Matrix(unique=False)
    # print(dm.values)
    # print(dm['timestamp'])

    # print(dm['timestamp'] == 200)
    # print(dm[dm['timestamp'] == 200])
    # print(dm['features'])
    # print(dm['features'] == 'high')
    # print(dm[dm['features'] == 'high'])
    # 从三维数据中查找轴名为code，元素为000001的矩阵切片
    axis_name = 'code'  # code timestamp
    compare_obj = '000001'  # '000001' 200
    # 打印数据全貌
    # print(dm)
    # 某个维度的轴信息
    # print(dm[axis_name])
    # 判断符合条件的信息
    # print(dm[axis_name] == compare_obj)
    # print(dm[dm[axis_name] == compare_obj])
    t1 = time.time()
    # print(dm)
    # 某个维度的轴信息
    # print(dm[axis_name])
    # 判断符合条件的信息
    # print(dm[axis_name] == compare_obj)
    # 返回判断切片
    dm_slice = dm[dm[axis_name] == compare_obj]
    df = dm[dm[axis_name] == compare_obj].to_pandas_df()  # 切片数据转为pd.DataFrame

    t2 = time.time()
    print('dm_slice',dm_slice)
    print(type(df))
    print(df)
    print('slice obj cost time:', t2 - t1)
