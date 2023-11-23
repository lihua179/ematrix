# -*- coding:utf-8 -*-
# Auther: Zed
"""
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
        3. Increase logic operation such as '>= <= > < !=/~ and & or / in' : dm[dm['datetime']>=20220105 and dm['code'] in ['000001','000002']]
        4. Increase more __magic_function__ such as __del__: del dm["feature=='close'"] / dm.delete('datetime<=20220105')
        5. Del dm[dm[::3]:ClassMatrix]
        6. Increase function fill_na: dm.fill_na('forward_value/backward_value/zero_value/avg_value')
        7. Query function: dm.query("20220105<=datetime<=20220506 and code in ['000001','000002'] and feature.capital>float(1e9)")
        -1. More simular pd.DataFrame operation function in Matrix
            * dm.sort(dm['feature']==close,ascend=True) / dm.sort('datetime')
            * dm.resort_index('datetime') / dm.resort_value('datetime.20220510')
                _datetime='datetime.20220510'.split('.')[1]
                it will check dtype of dm.axis_dtype of datetime : data=float(_datetime) if self.axis_dict['datetime'].dtype==float else str(_datetime)
"""
import time
from typing import Union, NewType
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle

set_print_limit = False  # 显示有限的输出设置

_SeriesType = Union[dict, tuple, list]

_SeriesTypeBase = [tuple, list]


def row_fill_na(data: pd.DataFrame, fill_list) -> pd.DataFrame:
    _na_data = np.full([len(fill_list), data.shape[1]], np.nan)

    new_data = np.vstack((data, _na_data))

    _index = list(data.index)
    _index.extend(fill_list)
    new_df = pd.DataFrame(new_data, index=_index, columns=data.columns)
    new_df.sort_index(inplace=True)

    return new_df


def columns_fill_na(data: pd.DataFrame, fill_list) -> pd.DataFrame:
    _na_data = np.full([data.shape[0], len(fill_list)], np.nan)
    new_data = np.hstack((data, _na_data))
    _columns = list(data.columns)
    _columns.extend(fill_list)
    new_df = pd.DataFrame(new_data, index=data.index, columns=_columns)

    return new_df


def diff_calcu(total_list, _array_list):
    _diff_list = list(set(total_list) - set(_array_list))
    return sorted(_diff_list)


def df_dict2matrix(df_dict, axis_name=None):
    if axis_name is None:
        axis_name = ['datetime','code', 'features']
    index = []
    row = []
    columns = []

    df_info_dict = {}
    for key, value in df_dict.items():
        index.append(key)

        row.extend(list(value.index.values))
        columns.extend(list(value.columns.values))
        df_info_dict[key] = {
            'row': list(value.index.values),
            'columns': list(value.columns.values),
        }
    values_dict = {}
    for key, value in deepcopy(df_info_dict).items():
        diff_row = diff_calcu(row, value['row'])
        diff_columns = diff_calcu(columns, value['columns'])

        if diff_row:

            new_df = row_fill_na(df_dict[key], diff_row)

            if diff_columns:
                new_df = columns_fill_na(new_df, diff_columns)
            new_df.sort_index(inplace=True)
            values_dict[key] = new_df
            continue
        if diff_columns:
            new_df = columns_fill_na(df_dict[key], diff_columns)
            new_df.sort_index(inplace=True)
            values_dict[key] = new_df
        else:
            df_dict[key].sort_index(inplace=True)
            values_dict[key] = df_dict[key]

    first_data_key = list(values_dict.keys())[0]
    first_data = values_dict[first_data_key]

    index = list(values_dict.keys())
    row = list(first_data.index)  # 时间记得排序
    columns = list(first_data.columns)

    axis_name = {axis_name[0]: index, axis_name[1]: row, axis_name[2]: columns}
    _array = np.empty([1, len(row), len(columns)])

    for key, value in values_dict.items():
        _array = np.vstack((_array, [value.values]))
    _array = _array[1:]
    # print(axis_name)
    return _array, axis_name


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


def check_Matrix_axis_length(matrix_data: np.ndarray, matrix_axis_name: dict):
    _shape = matrix_data.shape
    i = 0
    for key, value in matrix_axis_name.items():
        if len(value) != _shape[i]:
            raise ValueError(f'Error length! The axis_name length {_shape[i]}!= data length {len(value)}')
        i += 1


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


# 最基本的元素
class Unite:
    def __init__(self, value, name):
        self.value = value
        self.name = name

    def __str__(self):
        print(f'name:{self.name}\n', self.value)


class Series:
    def __init__(self, series: list, name: str = 'Undefined', unique=False, astype: str = None):

        self.dtype = object
        self.astype = astype

        self.name = name
        self._series = series
        self._series_value: np.ndarray
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
            # [('index_name0','value0'),('index_name1','value1')]
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

    def sort(self):
        pass

    @property
    def values(self):
        return self._series_value

    def _operator(self, sql):
        _axis_value = np.where(sql)[0]

        return Series(_axis_value, self.name, self.unique, 'loc.operator')
    def __len__(self):
        return len(self._series_value)
    #
    # def __contains__(self, other: list):
    #     # dm['timestamp'] in [1,2,3,4,5]
    #     # self.values[]

    def __or__(self, other):
        #     # or |

        return Series([self, other], name=self.name, astype='loc.or_series')

    def __ror__(self, other):
        # or |

        return Series([self, other], name=self.name, astype='loc.or_series')

    def __and__(self, other):

        return Series([self, other], name=self.name, astype='loc.and_series')

    def __rand__(self, other):

        return Series([self, other], name=self.name, astype='loc.and_series')

    def __eq__(self, other):
        if type(other) == list:
            res_list = []
            for i in range(len(other)):
                res = np.where(self.values == other[i])[0][0]
                res_list.append(res)

            return Series(res_list, name=self.name, astype='loc.operator')
        other = np.array(other)
        sql = self._series_value == other
        return self._operator(sql)

    def __lt__(self, other):
        sql = self._series_value < other
        return self._operator(sql)

    def __le__(self, other):
        sql = self._series_value <= other
        return self._operator(sql)

    def __gt__(self, other):
        sql = self._series_value > other
        return self._operator(sql)

    def __ge__(self, other):
        sql = self._series_value >= other
        return self._operator(sql)

    def __ne__(self, other):
        sql = self._series_value != other
        return self._operator(sql)

    def __str__(self):
        _data = _print(self._series)
        info = f'Name: {self.name}, Length:{self.length}, dtype:{self.dtype}'
        return _data + info

    def __repr__(self):
        return str(self)


_MatrixGetSingleItemType = Union[str, int, Series]
_MatrixGetMultiItemType = Union[str, int, Series]
_MatrixAxisNameType = Union[dict, list]
MatrixGetDataType = Union[np.ndarray, list, tuple, dict, None]


def matrix_to_pandas_df(matrix_np,axis_name) -> pd.DataFrame:
    _df = pd.DataFrame(matrix_np)
    _df.index = axis_name['index']
    _df.columns = axis_name['columns']
    return _df

class DataFrame:
    def __init__(self, matrix_np: np.ndarray = None, axis_name: dict = None, name='Undefined'):
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
        self.name = name
        # self=self.to_pandas_df()

    def to_pandas_df(self) -> pd.DataFrame:
        _df = pd.DataFrame(self.data)
        _df.index = self.index
        _df.columns = self.columns
        return _df

    def _print(self):
        pass

    def __str__(self):
        _df = self.to_pandas_df()
        return f'name:{self.name}\n' + str(_df)

    def __repr__(self):
        _df = self.to_pandas_df()
        return _df


_MatrixAppendObjType = Union[np.ndarray, pd.DataFrame]
_MatrixAppendAxisName = Union[str, int]


class Matrix:
    # 可接收多个pd.DataFrame聚合为Matrix数据类型
    # 标准样例：第一维度为时间，第二维度为合约标的代码，第三维度为特征因子features
    # 均需要外部提供，标准协议为：
    # axis_name={'timestamp':[xx,xx,xx],
    #            'code':[xx,xx,xx],
    #            'features':[xx,xx,xx],
    #           }
    # 需要解决一个问题：如果两个dataframe的长度不一致时如何解决
    def __init__(self, data: MatrixGetDataType = None, axis_name: _MatrixAxisNameType = None, unique=True, sort=False,
                 sort_axis=0, reverse=False):
        # unique如果能确保轴值是唯一的，那么可提高性能
        # sort=True：是否自动排序
        # sort_axis=0：自动排序默认轴
        # reverse：从小到大
        self._dimension_dict = {}  # 轴名为key，维度为value
        self._dimension_num_dict = {}  # 维度为key，轴名为value
        self._dimension_axis_dict = {'index': 0, 'row': 1, 'columns': 2}
        self._dimension_axis_num_dict = {0: 'index', 1: 'row', 2: 'columns'}
        self.unique = unique

        if type(data) == np.ndarray:
            data: np.ndarray
            if len(data.shape) < 3:
                raise ValueError(f'error data dimension:{len(data.shape)}')
        elif type(data) == dict:
            data, axis_name = df_dict2matrix(data, axis_name=axis_name)

        # list, tuple, dict 内部装着pd.dataframe

        elif type(data) != np.ndarray and not axis_name:
            # test data:
            axis_name = {'timestamp': [100, 200, 300, 400], 'code': ['000001', '000002', '000003'],
                         'features': ['close', 'high', 'low', 'open', 'ret']}
            data = np.random.random([4, 3, 5])

        check_Matrix_axis_length(data, axis_name)
        # elif demo:
        # axis_name = {'timestamp': [100, 200, 300, 400], 'code': ['000001', '000002', '000003'],
        #              'features': ['close', 'high', 'low', 'open', 'ret']}
        # data = np.random.random([4, 3, 5])
        # big data
        # axis_name = {'timestamp': list(range(500, 1500)), 'code': ['00000' + str(i) for i in range(2,5000)],
        #              'features': ['close', 'high', 'low', 'open',  'ret']}
        # data = np.random.random([1000, 4998, 5])

        self.data = data
        self.axis_name = axis_name

        self.index = None
        self.index_name = None
        self.row = None
        self.row_name = None
        self.columns = None
        self.columns_name = None

        self._axis_name_dict = {}
        self._axis_name_key = []
        self._axis_value_dtype = {}
        self._axis_value_num_dtype = {}
        self._is_sort = sort
        self._sort_axis = sort_axis
        self._reverse = reverse
        self.init()

    def T(self, axis_one, axis_two=None):
        if type(axis_one) == int and type(axis_two) == int:
            axis_one_num, axis_two_num = axis_one, axis_two

        elif type(axis_one) == str and type(axis_two) == str:
            axis_one_num, axis_two_num = self._dimension_dict[axis_one], self._dimension_dict[axis_two]

        elif type(axis_one) == list and not axis_two:
            if len(axis_one) == 3:
                if axis_one[0] == axis_one[1] or axis_one[1] == axis_one[2] or axis_one[0] == axis_one[2]:
                    raise ValueError(f'Error value, element should not be same value {axis_one}')
                if type(axis_one[0]) == str and type(axis_one[1]) == str and type(axis_one[2]) == str:
                    _one, _two, _three = self._dimension_dict[axis_one[0]], self._dimension_dict[axis_one[1]], \
                                         self._dimension_dict[axis_one[2]]
                    axis_one = [_one, _two, _three]
                elif type(axis_one[0]) == int and type(axis_one[1]) == int and type(axis_one[2]) == int:
                    _d = [0, 1, 2]
                    if axis_one[0] not in _d or axis_one[1] not in _d or axis_one[2] not in _d:
                        raise ValueError(f'Error value element in {axis_one}, it should be between in 0-2')

                else:
                    raise ValueError(f'Error type element in {axis_one}, it should be str or in')
                new_queue = axis_one
                self.data = self.data.transpose(new_queue)
                self._T(new_queue)
                return self
            else:
                raise ValueError(f"Error param length: {axis_one}, it's length should be 3")
        else:
            raise ValueError(
                f'Error type param: {axis_one, axis_two}, it should be double, int or str or axis_one is list and axis_two is None ')

        self.data = self.data.swapaxes(axis_one_num, axis_two_num)

        if axis_one_num == 0 and axis_two_num == 1 or axis_one_num == 1 and axis_two_num == 0:
            new_queue = [1, 0, 2]

        elif axis_one_num == 0 and axis_two_num == 2 or axis_one_num == 2 and axis_two_num == 0:
            new_queue = [2, 1, 0]
        elif axis_one_num == 1 and axis_two_num == 2 or axis_one_num == 2 and axis_two_num == 1:
            new_queue = [0, 2, 1]
        else:
            raise ValueError(f'error axis_num or axis_name: {axis_one, axis_two}')
        self._T(new_queue)

        return self

    def _T(self, new_queue):
        _dimension_dict = {}
        _axis_name_dict = {}
        for num in new_queue:
            _axis_name = self._dimension_num_dict[num]
            _axis_name_dict[_axis_name] = self.axis_name[_axis_name]
        self.axis_name = _axis_name_dict
        self.init()

    def copy(self):
        return deepcopy(self)

    def append(self, obj: _MatrixAppendObjType, name=None, axis_name=None):
        # axis_name:如果用户能输入轴名，那么就不需要作维度相等判断，因此性能会更高
        # 添加轴的轴value如datetime的value20220501
        # dm.append(pd.DataFrame,20220501,axis_name='datetime')
        if type(axis_name) == str:
            if axis_name in self._axis_name_key:

                _axis_dimension = self._dimension_dict[axis_name]
            else:
                raise ValueError(f'Not exist {axis_name} in {self._axis_name_key}')
        elif type(axis_name) == int and axis_name in [0, 1, 2]:
            if axis_name in [0, 1, 2]:
                _axis_dimension = axis_name
            else:
                raise ValueError(f'Not exist {axis_name} dimension in [0, 1, 2]')
        else:
            if not axis_name:
                _axis_dimension = axis_name = 0
            else:
                raise ValueError(f'Matrix not exist the axis_name: {axis_name}')

        if type(name) != self._axis_value_num_dtype[_axis_dimension]:
            raise TypeError(
                f"The type of {_axis_dimension} dimension's axis_value '{name}' is error, you should input type {self._axis_value_num_dtype[_axis_dimension]}")

        if name in self._axis_name_dict[self._dimension_num_dict[_axis_dimension]]:
            raise ValueError(f'Existed name {name} in axis')
        if _axis_dimension == 0:
            remain_axis_dimension = [1, 2]
        elif _axis_dimension == 1:
            remain_axis_dimension = [0, 2]
        else:
            remain_axis_dimension = [0, 1]
        if type(obj) == np.ndarray:
            obj = obj
        elif type(obj) == pd.DataFrame:
            obj = obj.values

        if obj.shape[0] == self.data.shape[remain_axis_dimension[0]] and obj.shape[1] == self.data.shape[
            remain_axis_dimension[1]]:
            obj = obj

        elif obj.shape[0] == self.data.shape[remain_axis_dimension[1]] and obj.shape[1] == self.data.shape[
            remain_axis_dimension[0]]:
            obj = obj.T

        else:
            raise ValueError(
                f'Error df_obj.shape:{obj.shape}!={remain_axis_dimension} or [{remain_axis_dimension[1], remain_axis_dimension[0]}]')
        if _axis_dimension == 0:
            obj = obj.reshape([1, obj.shape[0], obj.shape[1]])
        elif _axis_dimension == 1:
            obj = obj.reshape([obj.shape[0], 1, obj.shape[1]])
        else:
            obj = obj.reshape([obj.shape[0], obj.shape[1], 1])
        new_matrix = np.append(self.data, obj, axis=_axis_dimension)

        self.data = new_matrix
        _dimension_name = self._dimension_num_dict[_axis_dimension]

        self._axis_name_dict[_dimension_name].update({name: self.data.shape[_axis_dimension] - 1})
        self.axis_name[_dimension_name].append(name)

    def append_diff_df(self,df,name,*arg,**kwarg):
        if 'fill_na_axis' not in kwarg:
            kwarg['fill_na_axis']='row'
        obj_matrix=Matrix({name: df})
        return self.concat(obj_matrix,*arg,**kwarg)
    def concat(self, obj_matrix,fill_na_axis=None,*arg,**kwarg):
        # 拼接两个matrix数据
        # 自动确定相同的axis_name,第三条axis轴自然确定

        return concat(self, obj_matrix,fill_na_axis,*arg,**kwarg)

    def axis_value_type_check(self):
        # 检查下axis各轴值的类型
        # self._axis_value_dtype = {}
        i = 0
        for key, values in self.axis_name.items():
            self._axis_value_dtype[key] = type(values[0])
            self._axis_value_num_dtype[i] = type(values[0])
            i += 1
            for value in values[1:]:
                if type(value) != self._axis_value_dtype[key]:
                    raise TypeError(
                        f'The axis {key} type is not unified ({type(value)}!={self._axis_value_dtype[key]}')

    def init(self):
        if self.data.size == 0:
            return
        if len(self.data.shape) != 3:
            raise ValueError(f'error data shape {self.data}')
        if len(self.axis_name.keys()) != 3:
            raise ValueError(f'error axis_name len {self.data}')

        self._axis_name_to_range()

    def check_axis_data_len_match(self):
        # 检查轴长与数据shape是否相匹配
        axis_name_list=list(self.axis_name.keys())
        for i in range(len(axis_name_list)):
            if len(self.axis_name[axis_name_list[i]])!=self.data.shape[i]:
                raise ValueError(f"{axis_name_list[i]}'s length {len(self.axis_name[axis_name_list[i]])} != data.shape[{i}] {self.data.shape[i]}")
        pass
    def _axis_name_to_range(self):
        self.check_axis_data_len_match()
        self.axis_value_type_check()
        i = 0
        for key, value in self.axis_name.items():
            self._dimension_dict[key] = i
            self._dimension_num_dict[i] = key
            if i == 0:
                self.index = value
                self.index_name = key
            elif i == 1:
                self.row = value
                self.row_name = key
            else:
                self.columns = value
                self.columns_name = key
            i += 1

        _data_dict = {}
        for key, value in self.axis_name.items():
            # 用于实现类似loc精确定位矩阵元素
            _data_dict[key] = dict(zip(value, list(range(len(value)))))

        self._axis_name_dict = _data_dict
        self._axis_name_key = list(self._axis_name_dict.keys())
        if self._is_sort:
            # 是否自动排序
            self.sort(self._sort_axis, reverse=self._reverse)

    # ToDo 像df.loc一样定位数据
    @property
    def loc(self):
        return

    @property
    def shape(self):
        return self.data.shape


    # ToDo 像df.query一样条件查询数据
    def query(self):
        # res=matrix.query('100<=timestamp<=200 and close>21.5 or high>=100 and code=="000001.SZ"')
        #
        pass

    def fillna(self, method):
        # 沿着第一轴（通常认为是时间轴）前向或者后向填充空值
        # method 方法：前向，后向，先前向再后向，先后向再前向
        # 因为第三轴通常features特征数少于时间少于样本数，因此循环也是最少的

        # 增加跳过某些字段
        self.T(0, 2)

        if method == 'bfill':
            for i in range(self.data.shape[1]):
                self.data[:, i, :] = ffill(self.data[:, i, :])
        elif method == 'ffill':
            for i in range(self.data.shape[1]):
                self.data[:, i, :] = bfill(self.data[:, i, :])
        elif method == 'bffill':
            for i in range(self.data.shape[1]):
                self.data[:, i, :] = ffill(self.data[:, i, :])
                self.data[:, i, :] = bfill(self.data[:, i, :])
        elif method == 'fbfill':
            for i in range(self.data.shape[1]):
                self.data[:, i, :] = bfill(self.data[:, i, :])
                self.data[:, i, :] = ffill(self.data[:, i, :])
        else:
            raise ValueError(f'Error method: {method}')
        self.T(0, 2)
        return self

    def fillna_test(self, method, axis_name, axis_value=None):
        # 沿着第一轴（通常认为是时间轴）前向或者后向填充空值
        # method 方法：前向，后向，先前向再后向，先后向再前向
        # 因为第三轴通常features特征数少于时间少于样本数，因此循环也是最少的,这就是为什么要在开始前进行转置self.T(0, 2)

        # ToDo 目前是全量赋值，之后要做的就是给定指定字段进行赋值，如：axis_value='ask_v5',['ask_v4','ask_v5']

        if not axis_value:
            axis_value = slice(None, None, None)
        else:
            axis_value

        if axis_name == 'index' or axis_name == 0:

            if method == 'ffill':
                for i in range(self.data.shape[0]):
                    self.data[i,:,:] = bfill(self.data[i,:,:])


            elif method == 'bfill':
                for i in range(self.data.shape[0]):
                    self.data[i, :, :] = ffill(self.data[i, :, :])

            elif method == 'bffill':
                for i in range(self.data.shape[0]):
                    self.data[ i,:, :] = ffill(self.data[ i,:, :])
                    self.data[i,:,  :] = bfill(self.data[ i,:, :])
            elif method == 'fbfill':
                for i in range(self.data.shape[0]):
                    self.data[i, :, :] = bfill(self.data[i, :, :])
                    self.data[i, :, :] = ffill(self.data[i, :, :])

        elif axis_name == 'row' or axis_name == 1:

            self.T(0, 2)
            if method == 'ffill':
                for i in range(self.data.shape[1]):
                    self.data[:,i,  :] = bfill(self.data[ :,i, :])


            elif method == 'bfill':
                for i in range(self.data.shape[1]):
                    self.data[ :,i, :] = ffill(self.data[:,i,  :])

            elif method == 'bffill':
                for i in range(self.data.shape[1]):
                    self.data[ :,i, :] = ffill(self.data[:,i,  :])
                    self.data[ :,i, :] = bfill(self.data[ :,i, :])
            elif method == 'fbfill':
                for i in range(self.data.shape[1]):
                    self.data[ :,i, :] = bfill(self.data[:,i,  :])
                    self.data[:,i,  :] = ffill(self.data[ :,i, :])
            self.T(0, 2)
        elif axis_name == 'columns' or axis_name == 2:
            if method == 'ffill':
                for i in range(self.data.shape[2]):
                    self.data[:, :,i] = bfill(self.data[ :, :,i])

            elif method == 'bfill':
                for i in range(self.data.shape[2]):
                    self.data[:, :,i] = ffill(self.data[ :, :,i])

            elif method == 'bffill':
                for i in range(self.data.shape[2]):
                    self.data[:, :,i] = ffill(self.data[:, :,i])
                    self.data[:, :,i] = bfill(self.data[:, :,i])
            elif method == 'fbfill':
                for i in range(self.data.shape[2]):
                    self.data[:, :,i] = bfill(self.data[ :, :,i])
                    self.data[ :, :,i] = ffill(self.data[:, :,i])

    def fillna_pd(self,method,axis_name,axis_value=None):
        # 沿着第一轴（通常认为是时间轴）前向或者后向填充空值
        # method 方法：前向，后向，先前向再后向，先后向再前向
        # 因为第三轴通常features特征数少于时间少于样本数，因此循环也是最少的,这就是为什么要在开始前进行转置self.T(0, 2)

        # 通过pd fillna 则增加axis，axis_name两个参数
        if not axis_value:
            axis_value = slice(None, None, None)
        if axis_name=='index' or axis_name==0:
            columns=self.columns
            index = self.row
            # self.T(1, 2)
            if method == 'ffill':
                for i in range(self.data.shape[0]):

                    df=pd.DataFrame(self.data[i,:,:])
                    df.columns=columns
                    df.index = index
                    df[axis_value].fillna(method='bfill',inplace=True)
                    self.data[i,:,:] = df.values

            elif method == 'bfill':
                for i in range(self.data.shape[0]):

                    df=pd.DataFrame(self.data[i,:,:])
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='ffill',inplace=True)
                    self.data[i,:,:] = df.values

            elif method == 'bffill':
                for i in range(self.data.shape[0]):

                    df=pd.DataFrame(self.data[i,:,:])
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='ffill',inplace=True)
                    df[axis_value].fillna(method='bfill', inplace=True)
                    self.data[i,:,:] = df.values
            elif method == 'fbfill':
                for i in range(self.data.shape[0]):

                    df=pd.DataFrame(self.data[i,:,:])
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='bfill',inplace=True)
                    df[axis_value].fillna(method='ffill', inplace=True)
                    self.data[i,:,:] = df.values
            # self.T(1, 2)
        elif axis_name=='row'  or axis_name==1:
            columns = self.columns
            index = self.index
            self.T(0, 2)
            if method == 'ffill':
                for i in range(self.data.shape[1]):
                    df=pd.DataFrame(self.data[:, i, :].T)
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='bfill',inplace=True)
                    self.data[:, i, :] = df.values.T
            elif method == 'bfill':
                for i in range(self.data.shape[1]):

                    df=pd.DataFrame(self.data[:, i, :].T)
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='ffill',inplace=True)
                    self.data[:, i, :] = df.values.T
            elif method == 'bffill':
                for i in range(self.data.shape[1]):

                    df=pd.DataFrame(self.data[:, i, :].T)
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='ffill',inplace=True)
                    df[axis_value].fillna(method='bfill', inplace=True)
                    self.data[:, i, :] = df.values.T
            elif method == 'fbfill':
                for i in range(self.data.shape[1]):

                    df=pd.DataFrame(self.data[:, i, :].T)
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='bfill',inplace=True)
                    df[axis_value].fillna(method='ffill', inplace=True)
                    self.data[:, i, :] = df.values.T
            self.T(0, 2)
        elif axis_name=='columns'  or axis_name==2:
            columns = self.index
            index = self.row
            # self.T(0, 1)
            if method == 'ffill':
                for i in range(self.data.shape[2]):

                    df=pd.DataFrame(self.data[:,:,i])
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='bfill',inplace=True)
                    self.data[:,:,i] = df.values
            elif method == 'bfill':
                for i in range(self.data.shape[2]):

                    df=pd.DataFrame(self.data[:,:,i])
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='ffill',inplace=True)
                    self.data[:,:,i] = df.values
            elif method == 'bffill':
                for i in range(self.data.shape[2]):

                    df=pd.DataFrame(self.data[:,:,i])
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='ffill',inplace=True)
                    df[axis_value].fillna(method='bfill', inplace=True)
                    self.data[:,:,i] = df.values
            elif method == 'fbfill':
                for i in range(self.data.shape[2]):
                    df=pd.DataFrame(self.data[:,:,i])
                    df.columns = columns
                    df.index = index
                    df[axis_value].fillna(method='bfill',inplace=True)
                    df[axis_value].fillna(method='ffill', inplace=True)
                    self.data[:,:,i] = df.values
            # self.T(0, 1)

    def fillna_pd_test(self, method):
        # 沿着第一轴（通常认为是时间轴）前向或者后向填充空值
        # method 方法：前向，后向，先前向再后向，先后向再前向
        # 因为第三轴通常features特征数少于时间少于样本数，因此循环也是最少的
        self.T(0, 2)
        if method == 'ffill':
            for i in range(self.data.shape[1]):

                df=pd.DataFrame(self.data[:, i, :].T)
                df.fillna(method='bfill',inplace=True)
                self.data[:, i, :] = df.values.T
        elif method == 'bfill':
            for i in range(self.data.shape[1]):
                df = pd.DataFrame(self.data[:, i, :].T)
                df.fillna(method='ffill', inplace=True)
                self.data[:, i, :] = df.values.T
        elif method == 'bffill':
            for i in range(self.data.shape[1]):
                df = pd.DataFrame(self.data[:, i, :].T)
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                self.data[:, i, :] = df.values.T

        elif method == 'fbfill':
            for i in range(self.data.shape[1]):
                df = pd.DataFrame(self.data[:, i, :].T)
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                self.data[:, i, :] = df.values.T
        else:
            raise ValueError(f'Error method: {method}')
        self.T(0, 2)
        return self

    def sort(self, sort_axis_name: Union[str, int], assign_list: list = None, reverse=False):
        # 根据axis_name对应的轴值，进行二维矩阵的排序,可以根据指定的顺序列表assign_list（代替轴值）进行排序，
        # 可配合Series和features进行元素值排序，如根据某一天datetime的features.close的大小进行排序
        # 排序完后，dm.axis_name和dm._axis_name_dict需要重新赋值
        # reverse=False表示升序
        if type(sort_axis_name) == int:
            if sort_axis_name not in [0, 1, 2]:
                raise ValueError(f'Not exist {sort_axis_name} in [0,1,2]')
            _axis_value = self.axis_name[self._dimension_num_dict[sort_axis_name]]
            _dimension = sort_axis_name
            _name = self._dimension_num_dict[sort_axis_name]
        elif type(sort_axis_name) == str:
            if sort_axis_name not in self._axis_name_key:
                raise ValueError(f'Not exist {sort_axis_name} in {self._axis_name_key}')
            _axis_value = self.axis_name[sort_axis_name]
            _dimension = self._dimension_dict[sort_axis_name]
            _name = sort_axis_name
        else:
            raise TypeError(f'Error type sort_axis_name: {sort_axis_name}')

        if assign_list:
            _axis_value = assign_list
        else:
            _axis_value.sort(reverse=reverse)

        sort_index = [self._axis_name_dict[_name][sort_value] for sort_value in _axis_value]

        if _dimension == 0:
            self.data = self.data[sort_index, :, :]
        elif _dimension == 1:
            self.data = self.data[:, sort_index, :]
        else:
            self.data = self.data[:, :, sort_index]
        self.axis_name[_name] = _axis_value
        self._axis_name_dict[_name] = dict(zip(_axis_value, list(range(len(_axis_value)))))

    def sort_value(self, ):
        # dm.sort_value(dm[['datetime'==20220501,'futures'=='close']])
        # 根据两个维度确定的一条Series的值大小进行排序，比如datetime=20220501，futures=close，则确定20220501当天close从大到小进行排序
        pass

    def sort_index(self, reverse=False):
        # 根据轴值排序
        pass

    def sort_row(self, reverse=False):
        # 根据轴值排序
        pass

    def sort_columns(self, reverse=False):
        # 根据轴值排序
        pass

    def set_index(self, index_list=None):
        # self.index = index_list
        # 找到index对应的字符名字
        _axis_name=list(self.axis_name.keys())[0]
        self.axis_name[_axis_name]=index_list
        self._axis_name_to_range()


    def set_row(self, row_list=None):
        _axis_name = list(self.axis_name.keys())[1]
        self.axis_name[_axis_name] = row_list
        self._axis_name_to_range()

    def set_columns(self, columns_list=None):
        _axis_name = list(self.axis_name.keys())[2]
        self.axis_name[_axis_name] = columns_list
        self._axis_name_to_range()

    def __getitem__(self, *item):

        if len(item) == 1:
            if type(item[0]) in [str, int, slice, Series]:
                item = item[0]
                item: _MatrixGetSingleItemType
                if type(item) == str:
                    # dm['data_xx']:返回某维度所有index
                    if item not in self._axis_name_key:
                        raise ValueError(f'{item} not exist in Matrix')

                    return Series(list(self._axis_name_dict[item].keys()), item, self.unique)
                elif type(item) == Series:

                    # 精确区间定位
                    if item.dtype == 'loc.operator':
                        _dimension = self._dimension_dict[item.name]

                        _axis_name = deepcopy(self.axis_name)
                        _axis_name[item.name] = list(np.array(self.axis_name[item.name])[item.values])

                        if _dimension == 0:
                            _data = self.data[item.values, :, :]
                        elif _dimension == 1:
                            _data = self.data[:, item.values, :]
                        else:
                            _data = self.data[:, :, item.values]
                        if 1 in _data.shape:
                            return Matrix(_data, _axis_name, unique=self.unique).to_std_df()
                        return Matrix(_data, _axis_name, unique=self.unique)

                        # 精确区间带步长定位
                        # elif len(item.values) == 3:
                        #     pass

                    elif item.dtype == 'loc.in_series':

                        # loc_in类型：适用于如dm[dm['code'] in [000001,000002,...,]]的
                        # __in__ 定位指定样本
                        # 这一步只是作为bool类型的代替，提高效率的方案
                        _dimension = self._dimension_dict[item.index[0]]
                        _axis_name = deepcopy(self.axis_name)
                        _axis_name[item.name] = list(np.array(_axis_name[item.name])[item.values])
                        if _dimension == 0:
                            _data = self.data[item.values, :, :]
                        elif _dimension == 1:
                            _data = self.data[:, item.values, :]
                        else:
                            _data = self.data[:, :, item.values]

                        return Matrix(_data, _axis_name, unique=self.unique)

                    elif item.dtype == 'loc.and_series':

                        new_matrix = self
                        for _series in item.values:
                            if new_matrix.values.any():
                                new_matrix = new_matrix[_series]
                            else:
                                return
                        return new_matrix
                        # 多组Series 的bool类型数据，如:
                        #   0 Series([True,False]),name=code,dtype=bool
                        #   1 Series([1234567,1234568,...]),name=timestamp,dtype=loc
                        #  此时需要循环一步步定位，过滤数据
                        # 常出现在dm[dm['timestamp']>1234567 and dm['code'] in ['000001','000002',...] or feature='close']使用场景
                        # 因为and or in 的比较值Series DataFrame Matrix都有可能，所以都需要进行编写相关魔法函数
                        #

                    elif item.dtype == 'loc.or_series':

                        for _series in item.values:

                            new_matrix = deepcopy(self)

                            new_matrix = new_matrix[_series]
                            if new_matrix.values.any():
                                return new_matrix
                elif type(item) == slice:
                    _data = self.values[item]

                    _axis_name = deepcopy(self.axis_name)
                    _axis_name_key = self._dimension_num_dict[0]
                    _axis_name[_axis_name_key] = list(np.array(_axis_name[_axis_name_key])[item])

                    return Matrix(_data, _axis_name, self.unique)

                elif type(item) == int:

                    _data = self.values[item]

                    _axis_name = deepcopy(self.axis_name)
                    _axis_name.pop(self._dimension_num_dict[0])
                    i = 0
                    _df_name = {'index': [], 'columns': []}
                    for key, value in _axis_name.items():
                        if i == 0:
                            _df_name['index'] = value
                            i += 1
                        else:
                            _df_name['columns'] = value
                    # print(item)
                    # print(self._dimension_num_dict[0])
                    # print(self.axis_name)
                    # print(self.axis_name[self._dimension_num_dict[0]])

                    return matrix_to_pandas_df(_data,_df_name)
                    # return DataFrame(_data, _df_name, name=str(self._dimension_num_dict[0]) + '==' + str(
                    #     self.axis_name[self._dimension_num_dict[0]][item]))

            elif len(item[0]) == 2:
                if type(item[0][0]) == str:

                    # support: dm['code',[000001,000002]], dm['code',1:10:2], dm['code',5]
                    if item[0][0] in ['index', 'row', 'columns']:
                        _dimension = self._dimension_axis_dict[item[0][0]]
                    else:
                        _dimension = self._dimension_dict[item[0][0]]
                    _name = self._dimension_num_dict[_dimension]
                    if type(item[0][1]) == slice:
                        select_list = item[0][1]
                    elif type(item[0][1]) == list:

                        _axis_value = self.axis_name[_name]
                        axis_value_list = []
                        for i in range(len(item[0][1])):
                            _res = np.where(np.array(_axis_value) == item[0][1][i])[0][0]
                            axis_value_list.append(_res)
                        select_list = axis_value_list
                    elif type(item[0][1]) == int:
                        select_list = item[0][1]
                    else:
                        # self._dimension_dict
                        # self._dimension_num_dict
                        # self._axis_name_dict
                        # self._axis_name_key
                        raise ValueError(f'Error type item{type(item[0][1])}')
                    if _dimension == 0:
                        _value = self.values[select_list, :, :]
                    elif _dimension == 1:
                        _value = self.values[:, select_list, :]
                    else:
                        _value = self.values[:, :, select_list]

                    if type(item[0][1]) == int:
                        _axis_name = deepcopy(self.axis_name)
                        _axis_name.pop(_name)
                        new_axis = {}
                        _i = 0
                        # print(_axis_name)
                        for _key0, _value0 in _axis_name.items():
                            if _i == 0:
                                new_axis['index'] = _value0
                                _i += 1
                            else:
                                new_axis['columns'] = _value0

                        return matrix_to_pandas_df(_data, _df_name)
                        # return DataFrame(_value, axis_name=new_axis,
                        #                  name=_name + '==' + str(self.axis_name[_name][select_list]))
                    _axis_value = self.axis_name[_name]
                    _axis_value = np.array(_axis_value)

                    _axis_value = list(_axis_value[select_list])
                    self.axis_name[_name] = _axis_value

                    return Matrix(_value, deepcopy(self.axis_name), unique=self.unique)

                elif type(item[0][0]) == int:
                    if type(item[0][1]) == int:
                        raise ValueError(f'Unsupported at least 2 int type params: {item[0]}')
                        # _value = self.values[item[0][0], type(item[0][1])]
                        # print(_value)
                    elif type(item[0][1]) == slice:
                        # ToDO 要么只有slice，那么只有一个int
                        raise ValueError(f'Unsupported 1 int type  and 1 slice type params: {item[0]}')
                    else:
                        raise ValueError(f'Unsupported  params: {item[0]}')

                elif type(item[0][0]) == slice:

                    if type(item[0][1]) == int:
                        raise ValueError(f'Unsupported  params: {item[0]}')
                    elif type(item[0][1]) == slice:
                        # 支持双切割 dm[2:10:1,5:10:2]
                        _value = self.data[item[0][0], item[0][1]]
                        i = 0
                        _axis_name = {}
                        _axis_name_copy = deepcopy(self.axis_name)

                        for key, value in _axis_name_copy.items():
                            if i == 0:
                                _axis_name[key] = list(np.array(value)[item[0][0]])

                            elif i == 1:
                                _axis_name[key] = list(np.array(value)[item[0][1]])
                            else:
                                _axis_name[key] = list(np.array(value))
                            i += 1

                        return Matrix(_value, _axis_name, unique=self.unique)
                    else:
                        raise ValueError(f'Unsupported  params: {item[0]}')


                else:
                    raise ValueError(f'Unsupported  params: {item[0]}')

            elif len(item[0]) == 3:
                if type(item[0][0]) == slice and type(item[0][1]) == slice and type(item[0][2]) == slice:
                    _value = self.data[item[0][0], item[0][1], item[0][2]]
                    i = 0
                    _axis_name = {}
                    _axis_name_copy = deepcopy(self.axis_name)

                    for key, value in _axis_name_copy.items():
                        if i == 0:
                            _axis_name[key] = list(np.array(value)[item[0][0]])

                        elif i == 1:
                            _axis_name[key] = list(np.array(value)[item[0][1]])
                        else:
                            _axis_name[key] = list(np.array(value)[item[0][2]])
                        i += 1

                    return Matrix(_value, _axis_name, unique=self.unique)

                else:
                    raise ValueError(f'Unsupported  params: {item[0]}')

        else:
            raise ValueError(f'Unsupported  params: {item[0]}')
            # print('_MatrixGetMultiItemType')
            # pass

    def to_datetime64(self,axis_name=None):
        if not axis_name:
            axis_name = list(self.axis_name.keys())[0]
        return to_datetime64(self.axis_name[axis_name])

    def to_timestamp(self,axis_name=None):
        if not axis_name:
            axis_name = list(self.axis_name.keys())[0]
        return to_timestamp(self.axis_name[axis_name])

    def astype(self, axis_name=None, element=None, type=np.float64):
        # 对某一面进行数据类型转换
        if not axis_name and not element:
            self.data.astype(type)
        return self
        # print(type(res0[0]))

    def to_pkl(self, save_name: str):
        to_pkl(self, save_name)

    def to_df(self, is_matrix_index=True, matrix_index_split='.'):

        return to_pandas(self, is_matrix_index=is_matrix_index, matrix_index_split=matrix_index_split)

    def to_std_df(self):
        return to_std_pandas(self)

    @property
    def values(self):
        return self.data

    def _print(self):

        _axis_name_list = []
        _reverse_dimension = {v: k for k, v in self._dimension_dict.items()}
        columns_str = list(self._axis_name_dict[_reverse_dimension[2]].keys())
        if len(columns_str) > 10:
            columns_str = str(columns_str[:2])[:-1] + '...' + str(columns_str[-2:])[1:]
        _print_data = "\n             " + str(columns_str)

        for i, value in enumerate(self.data):
            if i >= 2:
                last_count = list(self._axis_name_dict[_reverse_dimension[0]].keys())[-1]
                last_row = list(self._axis_name_dict[_reverse_dimension[1]].keys())[-1]
                last_data = self.data[-1][-1]
                if len(last_data) >= 5:
                    last_data = str(last_data[:2])[:-1] + '...' + str(last_data[-2:])[1:]

                _print_data += '                  ......\n' + str(last_count) + '\n' + '...\n' + str(
                    last_row) + '  ' + str(last_data)
                break

            _p = list(self._axis_name_dict[_reverse_dimension[0]].keys())[i]
            _print_data += '\n' + str(_p) + '\n'
            for j in range(len(value)):
                if j >= 2:
                    _print_data += '                  ......\n'
                    break

                _print_data += str(list(self._axis_name_dict[_reverse_dimension[1]].keys())[j])
                _value = value[j]
                # 控制列
                if len(_value) >= 5:
                    _value = str(_value[:2])[:-1] + '...' + str(_value[-2:])[1:]
                _print_data += '  ' + str(_value) + '\n'

        _matrix_info = f'\nIndex: {self._axis_name_key[0]}, Row: {self._axis_name_key[1]}, Columns: {self._axis_name_key[2]}, Shape: {self.data.shape}, dtype: matrix'
        _print_data += _matrix_info + '\n'
        return _print_data

    def __delitem__(self, key):
        #  del dm[1:5:2]
        #  ToDo del dm['features','datetime'] 确定dm['features']中‘datetime’的排序，np删除该面
        if type(key) == slice:
            axis_name = np.array(self.axis_name[self._dimension_num_dict[0]])
            axis_name = np.delete(axis_name, key)
            self.axis_name[self._dimension_num_dict[0]] = list(axis_name)
            self.data = np.delete(self.data, key, 0)
            self._axis_name_to_range()

    def pop(self,axis_name, element):
        self.delet(axis_name, element)
        return self

    def delet(self, axis_name, element):
        if type(axis_name) == str:
            key_list = self.axis_name[axis_name]
            # print(self._dimension_dict)
            dimesion = self._dimension_dict[axis_name]
            _index = key_list.index(element)
            key_list.pop(_index)
            self.data = np.delete(self.data, _index, axis=dimesion)
            self._axis_name_to_range()
        return self

    # def __delslice__(self, i, j):
    #
    #
    #     pass

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        if self.data.size == 0:
            return 'None'
        return self._print()

    def __repr__(self):
        return str(self)


_FillNaType = Union[str, int]


def diff_axis_check(matrix_obj: Matrix, axis_name: str = None):
    # 知道其中一个或者两个的轴名，返回剩下轴名
    return list(set(matrix_obj._axis_name_key) - {axis_name})


def matrix_fill_na(matrix_obj: Matrix, axis_name: _FillNaType = None, name='UnDefinedName'):
    # 三维矩阵在某一维度上增加一块na二维矩阵，用于在concat前进行统一另外两个维度
    # name：添加矩阵元素的键名
    # axis_name：指定轴名添加na矩阵
    if type(axis_name) == str:
        if axis_name in matrix_obj._axis_name_key:

            _dimension = matrix_obj._dimension_dict[axis_name]
        else:
            raise TypeError(f'Error type of {axis_name}')

    elif type(axis_name) == int:
        # int 为轴维度
        if axis_name in [0, 1, 2]:
            _dimension = axis_name

        else:
            raise ValueError(f'Error value of {axis_name}')

    else:
        raise TypeError(f'Error type of {axis_name}')

    _shape = matrix_obj.values.shape
    if _dimension == 0:
        _na_matrix_shape = [_shape[1], _shape[2]]
    elif _dimension == 1:
        _na_matrix_shape = [_shape[0], _shape[2]]
    else:
        _na_matrix_shape = [_shape[0], _shape[1]]
    _na_matrix = np.full(_na_matrix_shape, np.nan)

    matrix_obj.append(_na_matrix, name, axis_name=_dimension)

    return matrix_obj


_ConcatFillNaType = Union[str, list]


def _matrix_fill_na(matrix0, matrix1, fill_na_axis):
    for _axis_name in fill_na_axis:
        if _axis_name == 'index':
            _axis_value = 0
            _matrix0_axis_value = matrix0.index
            _matrix1_axis_value = matrix1.index

        elif _axis_name == 'row':
            _axis_value = 1
            _matrix0_axis_value = matrix0.row
            _matrix1_axis_value = matrix1.row
        else:
            _axis_value = 2
            _matrix0_axis_value = matrix0.columns
            _matrix1_axis_value = matrix1.columns

        matrix1_diff_value = [_matrix0 for _matrix0 in _matrix0_axis_value if
                              _matrix0 not in _matrix1_axis_value]
        matrix0_diff_value = [_matrix1 for _matrix1 in _matrix1_axis_value if
                              _matrix1 not in _matrix0_axis_value]

        if matrix0_diff_value:

            for _diff_value in matrix0_diff_value:
                matrix0 = matrix_fill_na(matrix0, _axis_value, _diff_value)

            matrix0.sort(sort_axis_name=_axis_value)
        if matrix1_diff_value:
            for _diff_value in matrix1_diff_value:
                matrix1 = matrix_fill_na(matrix1, _axis_value, _diff_value)

            matrix1.sort(sort_axis_name=_axis_value)

    return matrix0, matrix1


def concat(matrix0: Matrix, matrix1: Matrix, fill_na_axis: _ConcatFillNaType = None,*arg,**kwarg) -> Matrix:
    # 拼接两个matrix数据
    # 自动确定相同的axis_name以及长度,第三条axis轴自然确定
    # 因为是自动拼接，因此数据要求其中两个维度等长，第三个元素完全不相等
    # 当要求等长的两个维度其中一个不等或者两个都不等时：
    # fill_na_axis:指定自动填充na的维度index，row，columns,使该维度轴值完全相同
    """
    demo:

    new_dm = concat(dm0, dm1, ['row','columns'])

    new_dm = concat(dm0, dm1, 'row')

    """
    if fill_na_axis:
        if type(fill_na_axis) == str:
            if fill_na_axis not in ['index', 'row', 'columns']:
                raise ValueError(f'Error value fill_na_axis: {fill_na_axis}')
            matrix0, matrix1 = _matrix_fill_na(matrix0, matrix1, [fill_na_axis])


        elif type(fill_na_axis) == list:
            if fill_na_axis not in [['index', 'row', 'columns'],
                                    ['row', 'columns'],
                                    ['index', 'columns'],
                                    ['index', 'row'],
                                    ]:
                raise ValueError(f'Error value fill_na_axis: {fill_na_axis}')
            matrix0, matrix1 = _matrix_fill_na(matrix0, matrix1, fill_na_axis)


        else:
            raise TypeError(f'Error type of fill_na_axis: {fill_na_axis}')
    # print('res__________________')
    # print('data',matrix0.values, matrix1.values)
    matrix0_axis_name_dict = matrix0._axis_name_dict
    matrix0_axis_name_keys = list(matrix0_axis_name_dict.keys())

    matrix0_axis_name_values = list(matrix0.axis_name.values())
    matrix1_axis_name_dict = matrix1._axis_name_dict
    matrix1_axis_name_keys = list(matrix1_axis_name_dict.keys())

    matrix1_axis_name_values = list(matrix1.axis_name.values())

    same_axis_value = []
    same_axis_keys = []
    diff_axis_name = ''
    diff_axis_value = {}

    for i in range(len(matrix0_axis_name_values)):

        if matrix0_axis_name_values[i] == matrix1_axis_name_values[i]:
            same_axis_value.append(matrix0_axis_name_values[i])
            same_axis_keys.append(matrix0_axis_name_keys[i])
        else:
            diff_axis_name = matrix0_axis_name_keys[i]
            diff_axis_value[0] = matrix0_axis_name_values[i]
            diff_axis_value[1] = matrix1_axis_name_values[i]

    if matrix0_axis_name_keys != matrix1_axis_name_keys:
        raise ValueError(f"matrix0's {matrix0_axis_name_keys} != matrix1's {matrix0_axis_name_keys}")

    if len(same_axis_value) != 2:
        raise ValueError(
            f"matrix0 and matrix1 same_axis_value must be 2 ,but you have {len(same_axis_value)} (diff axis: {diff_axis_value})")

    if list(set(diff_axis_value[0]) & set(diff_axis_value[1])):
        raise ValueError(
            f'Same axis_value matrix0 and axis_matrix1 in {diff_axis_name}:({diff_axis_value[0]}'
            f' and {diff_axis_value[1]}), you should keep different')

    for i in range(len(matrix0_axis_name_keys)):
        if matrix0_axis_name_dict[matrix0_axis_name_keys[i]] in same_axis_value:
            same_axis_keys.append(matrix0_axis_name_keys[i])

    if matrix0._dimension_dict[diff_axis_name] != matrix1._dimension_dict[diff_axis_name]:
        raise ValueError(f"Matrix0's {diff_axis_name} dimension != matrix1's {diff_axis_name} dimension")
    if matrix0._dimension_dict[same_axis_keys[0]] != matrix1._dimension_dict[same_axis_keys[0]]:
        raise ValueError(f"Matrix0's {same_axis_keys[0]} dimension != matrix1's {same_axis_keys[0]} dimension")
    if matrix0._dimension_dict[same_axis_keys[1]] != matrix1._dimension_dict[same_axis_keys[1]]:
        raise ValueError(f"Matrix0's {same_axis_keys[1]} dimension != matrix1's {same_axis_keys[1]} dimension")

    new_matrix_value = np.concatenate((matrix0.values, matrix1.values), axis=matrix0._dimension_dict[diff_axis_name])

    new_diff_axis = matrix0.axis_name[diff_axis_name] + matrix1.axis_name[diff_axis_name]
    _axis_name = {}
    for i in range(len(matrix0_axis_name_keys)):
        if matrix0_axis_name_keys[i] == diff_axis_name:
            _axis_name.update({diff_axis_name: new_diff_axis})
        else:
            _axis_name.update({matrix0_axis_name_keys[i]: matrix0.axis_name[matrix0_axis_name_keys[i]]})

    return Matrix(new_matrix_value, _axis_name, True,*arg,**kwarg)


def to_pkl(obj: Matrix, dir_name):
    with open(dir_name, 'wb') as f:  # 打开文件
        pickle.dump((obj.data, obj.axis_name, obj.unique), f)  # 用 dump 函数将 Python 对象转成二进制对象文件


def read_pkl(dir_name)->Matrix:
    with open(dir_name, 'rb') as f:  # 打开文件
        _obj = pickle.load(f)  # 用 dump 函数将 Python 对象转成二进制对象文件
    return Matrix(_obj[0], _obj[1], _obj[2])


def read_pandas(obj: pd.DataFrame, shape=None, axis_name: dict = None):
    dm_value = obj.values.reshape(shape)
    dm = Matrix(dm_value, axis_name)
    return dm


def to_pandas(obj: Matrix, is_matrix_index: bool = True, matrix_index_split: str = '.') -> pd.DataFrame:
    # 直接将三维转为标准的二维df
    # 检查哪个是一条数据，就能确定
    new_dm_data = obj.data.reshape([obj.data.shape[0] * obj.data.shape[1], obj.data.shape[2]])
    _index = obj.index
    _row = obj.row
    _columns = obj.columns

    _index_same_list = []
    _index_list = []
    for i in range(len(_index)):
        _index_same_list.extend([_index[i] for j in range(len(_row))])
    row_list = _row * len(_index)
    new_array = np.append(np.array([_index_same_list]), np.array([row_list]), axis=0).T
    new_array = np.append(new_array, np.array(new_dm_data), axis=1)

    if is_matrix_index:
        index = [str(_index_same_list[i]) + matrix_index_split + str(row_list[i]) for i in range(len(_index_same_list))]
    else:
        index = list(range(new_array.shape[0]))
    columns = [obj.index_name, obj.row_name]
    columns.extend(obj.columns)
    df = pd.DataFrame(new_array, index=index, columns=columns)

    return df

def to_datetime64(datetime_list)->np.ndarray:
    #
    return np.array(list(map(np.datetime64, datetime_list)))
def to_timestamp(datetime_list)->np.ndarray:
    return np.array(datetime_list).astype(np.float64)
def to_std_pandas(obj: Matrix) -> pd.DataFrame:
    _axis_name_value = list(obj.axis_name.values())
    _axis_name_key = list(obj.axis_name.keys())
    _dimension = -1
    for i in range(len(_axis_name_value)):
        if len(_axis_name_value[i]) == 1:
            _dimension = i
    if _dimension != -1:
        if _dimension == 0:
            _data = obj.values[0, :, :]
        elif _dimension == 1:
            _data = obj.values[:, 0, :]
        elif _dimension == 2:
            _data = obj.values[:, :, 0]
        else:
            raise ValueError(f'Error dimension {_dimension}')

        _axis_name = obj.axis_name
        _axis_name.pop(_axis_name_key[_dimension])
        i = 0
        for key, value in deepcopy(_axis_name).items():
            if i == 0:
                _axis_name['index'] = value
                i += 1
            else:
                _axis_name['columns'] = value
        return DataFrame(_data, _axis_name, _axis_name_key[_dimension]).to_pandas_df()


def ffill(arr: np.ndarray):
    arr = arr.astype(np.float64)
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def bfill(arr):
    arr = arr.astype(np.float64)
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out

def check_data_type(data: np.ndarray):
    data_type = []

    check_columns = []
    for i, value in enumerate(data[0]):
        if value:
            if np.isnan(value):
                check_columns.append(i)
                continue
            data_type.append((i, type(value)))

    _row = 1
    for z in range(1, len(data)):
        del_list = []
        for j in range(len(check_columns)):
            i = check_columns[j]
            value = data[_row][i]
            if value:
                if np.isnan(value):
                    # check_columns.append(i)
                    continue
                del_list.append(i)
                data_type.append((i, type(value)))
        for del_ in del_list:
            check_columns.pop(check_columns.index(del_))
        _row += 1
        if not check_columns:
            break
    if check_columns:
        for i in range(len(check_columns)):
            data_type.append((check_columns[i], 'np.nan'))
    return data_type
if __name__ == '__main__':
    dm = Matrix()
    print(dm)
    dm.delet('features', 'close')
    print(dm)
