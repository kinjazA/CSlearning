# 01 文件的读取和写入

## 01.1 文件读取

```python
# 导入Pandas库
import pandas as pd
```

> * 常见的文件类型主要有`csv`、`txt`、`excel`
> * `csv`、`txt` 和`excel` 文件分别可以用`read_csv()`、`read_table()`和`read_excel()`读取，==其中的第一个传入参数均为相应文件的绝对路径或相对路径==

```python
# 例子
df_csv = pd.read_csv('data/ch2/my_csv.csv')
df_txt = pd.read_table('data/ch2/my_table.txt')
df_excel = pd.read_excel('data/ch2/my_excel.xlsx')
```

### 01.1.1 常用函数参数

> * 这些函数有一些公共参数，含义如下：将`header` 设置为`None` 表示第一行不作为列名
> * `parse_dates` 表示需要转化为时间的列
> * `index_col`表示把某一列或几列作为索引
> * `usecols`表示读取列的集合，默认读取所有列
> * `nrows`表示读取的数据行数

```python
pd.read_table('data/ch2/my_table.txt', header=None)
pd.read_csv('data/ch2/my_csv.csv', index_col=['col1', 'col2'])
pd.read_table('data/ch2/my_table.txt', usecols=['col1', 'col2'])
pd.read_csv('data/ch2/my_csv.csv', parse_dates=['col5'])
pd.read_excel('data/ch2/my_excel.xlsx', nrows=2)
```

> * 在读取`txt`文件时，经常会遇到分隔符非空格的情况，`read_table()`有一个分割参数`sep`，它使得用户可以自定义分割符号来进行对`txt`类型数据的读取，同时需要指定引擎（engine）为Python

```python
# 比方说是以||||作为分割的，要加转义字符就是
pd.read_table('data/ch2/my_table_special_sep.txt',sep= '\|\|\|\|',engine= 'python')
```

## 01.2 数据写入

> * 一般情况下，pandas 会在数据写入时包含当前的数据索引，但很多情况下我们并==不需要将默认的整数索引（0～n−1，n 为行数）包含到输出表中，此时我们可以把`index`设置为`False`==，该操作能在保存输出表的时候把索引去除

```python
df_csv.to_csv('data/ch2/my_csv_saved.csv', index=False)
df_excel.to_excel('data/ch2/my_excel_saved.xlsx', index=False)
```

> * `to_csv()`函数可以将数据保存为`txt`文件，并且允许自定义分隔符、常用制表符t 分割

```python
df_txt.to_csv('data/ch2/my_txt_saved.txt', sep='\t', index=False)
```

# 02 Pandas的基本数据结构

> * 有两种基本的数据结构，分别是存储一维值属性values 的`Series` 和存储二维值属性values 的`DataFrame`
> * 可以说`column`的每一列都是`series`

## 02.1 Series

> * 是一个一维标记数组，能够保存任何数据类型（整数、字符串、浮点数、Python 对象等）。轴标签统称为**索引**

```python
# 基本用法
s = pd.Series(data, index=index)
```

> * `data`为字典时，`index`可以从字典实例化。如果传递了索引，则将拉出索引中标签对应的数据中的值

```python
In [9]: d = {"a": 0.0, "b": 1.0, "c": 2.0}

In [10]: pd.Series(d)
Out[10]: 
a    0.0
b    1.0
c    2.0
dtype: float64

In [11]: pd.Series(d, index=["b", "c", "d", "a"])
Out[11]: 
b    1.0
c    2.0
d    NaN
a    0.0
dtype: float64
```

> * `data`为ndarray时，`index`**的**长度必须与data 的长度相同。如果没有传递索引，则会创建一个值为`[0, ..., len(data) - 1]`的索引

```python
In [3]: s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

In [4]: s
Out[4]: 
a    0.469112
b   -0.282863
c   -1.509059
d   -1.135632
e    1.212112
dtype: float64

In [6]: pd.Series(np.random.randn(5))
Out[6]: 
0   -0.173215
1    0.119209
2   -1.044236
3   -0.861849
4   -2.104569
dtype: float64
```

> * `data`是标量值时，则必须提供`index`。该标量值将重复以匹配`index`的长度

```python
In [12]: pd.Series(5.0, index=["a", "b", "c", "d", "e"])
Out[12]: 
a    5.0
b    5.0
c    5.0
d    5.0
e    5.0
dtype: float64
```

### 02.1.1 常用属性

> * 常用的属性有`.values`（返回`series`的值列表）、`.index`（返回`series`的索引列表）、`.name`（返回`series`的名字）、`.shape`（返回`series`的长度）、`.size`（也是返回`series`的长度，即元素个数）

## 02.2 DataFrame

> * `DataFrame`在`Series`的基础上增加了列索引，可以把它理解为一种将一组具有公共索引的`Series`拼接而得到的数据
> * 与 Series 一样，DataFrame 接受许多不同类型的输入

```python
# 基本语法
>>> d = {'col1': [1, 2], 'col2': [3, 4]}
>>> df = pd.DataFrame(data=d)
>>> df
   col1  col2
0     1     3
1     2     4
```

> * 当`data`为`series`构成的字典时，生成的`index`将是各个系列`index`的并集。如果有任何嵌套字典，它们将首先转换为`series`。如果没有传递列，则列将是字典键的有序列表

```python
In [38]: d = {
   ....:     "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
   ....:     "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
   ....: }
   ....: 

In [39]: df = pd.DataFrame(d)

In [40]: df
Out[40]: 
   one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0

In [41]: pd.DataFrame(d, index=["d", "b", "a"])
Out[41]: 
   one  two
d  NaN  4.0
b  2.0  2.0
a  1.0  1.0

In [42]: pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"])
Out[42]: 
   two three
d  4.0   NaN
b  2.0   NaN
a  1.0   NaN
```

> * 当`data`为`ndarray`构成的字典时，所有`ndarray`必须具有相同的长度。如果传递`index`，它也必须与数组的长度相同。如果没有传递`index`，结果将为`range(n)` ，其中`n`是`ndarray`长度

```python
In [45]: d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}

In [46]: pd.DataFrame(d)
Out[46]: 
   one  two
0  1.0  4.0
1  2.0  3.0
2  3.0  2.0
3  4.0  1.0

In [47]: pd.DataFrame(d, index=["a", "b", "c", "d"])
Out[47]: 
   one  two
a  1.0  4.0
b  2.0  3.0
c  3.0  2.0
d  4.0  1.0
```

还有更多的细节，可以看[这个链接](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe)

> * 在`DataFrame`中可以用`[col_name]`与`[col_list]`来取出相应的列与由多个列组成新的`DataFrame`，结果分别为`Series`和`DataFrame`

```python
df['col_0']
df[['col_0', 'col_1']]
```

> * 使用`to_frame()`函数可以把序列转换为列数为1 的`DataFrame`

```python
df['col_0'].to_frame()
```

### 02.2.1 常用属性

> * `.to_numpy()`，会转成数组

```python
>>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
array([[1, 3],
       [2, 4]])
```

> * `.index`，获取`DataFrame`的索引（行标签）

```python
>>> df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Aritra'],
...                    'Age': [25, 30, 35],
...                    'Location': ['Seattle', 'New York', 'Kona']},
...                   index=([10, 20, 30]))
>>> df.index
Index([10, 20, 30], dtype='int64')

# 也可以赋值进去对其进行修改
>>> df.index = [100, 200, 300]
>>> df
    Name  Age Location
100  Alice   25  Seattle
200    Bob   30 New York
300  Aritra  35    Kona
```

> * ==`.columns`，获取`DataFrame `的列标签（一般来说就是第一行的标签）==
>
> * ==`.dtypes`，获取每列的数据类型==

```python
>>> df = pd.DataFrame({'float': [1.0],
...                    'int': [1],
...                    'datetime': [pd.Timestamp('20180310')],
...                    'string': ['foo']})
>>> df.dtypes
float              float64
int                  int64
datetime    datetime64[ns]
string              object
dtype: object
```

> * `.shape`，获取表示`DataFrame`维度的元组

```python
>>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4],
...                    'col3': [5, 6]})
>>> df.shape
(2, 3)
```

> * 通过`.T`可以把`DataFrame`的行列进行转置
> * ==当想删除某一个列或行时，可以使用`drop`方法，当`axis`取值为1 时删除列，而当`axis`取值为0 时删除行==

```python
df.drop(["col_3"], axis=1)
df.drop(["row_1"], axis=0)
```

> * ==`Series`或`DataFrame`的绝大多数方法在默认参数下都不会改变原表，而是返回一个临时拷贝。当真正需
>   要在`df`上删除时，使用赋值语句`df=df.drop(...)`即可==

# 03 常用函数

## 03.1 汇总函数

> * 查看表的前几行或后几行
> * 使用`head()`函数和`tail()`函数，它们分别返回表或者序列的前`n`行和后`n`行信息，其中`n`默认为5

```python
df.head(2)
df.tail(3)
```

> * 打印`DataFrame`的简洁摘要
> * `.info()`，输出的信息包括索引数据类型和列、非空值和内存使用情况
> * `.describe()`，生成描述性统计数据，包括总结分布的集中趋势、离散度和形状的统计数据，不包括`NaN`值

```python
df.info()
df.describe()
```

## 03.2 特征统计函数

> * 在`Series`和`DataFrame`上定义了许多统计函数，最常见的是`sum()`、`mean()`、`median()`、`var()`、`std()`、`max()`和`min()`

```python
>>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
>>> df
       a   b
tiger  1   2
zebra  2   3
>>> df.mean()
a   1.5
b   2.5
dtype: float64

# 参数axis用于指定按照行还是列计算，axis=0是按照列计算，不写也是按照列；axis=1是按照行来计算
>>> df.mean(axis=1)
tiger   1.5
zebra   2.5
dtype: float64

# 参数numeric_only=True使得只会对数值进行计算，避免报错
>>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},index=['tiger', 'zebra'])
>>> df.mean(numeric_only=True)
a   1.5
dtype: float64
```

> * 中位数方法：`quantile()`
> * 可以使用参数`method`和`interpolation`来组合实现对全部列进行计算
> * 详细可查看[pandas文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html#pandas.DataFrame.quantile)

```python
>>> df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
                      columns=['a', 'b'])
>>> df.quantile(.1)
a    1.3
b    3.7
Name: 0.1, dtype: float64
>>> df.quantile([.1, .5])
       a     b
0.1  1.3   3.7
0.5  2.5  55.0
```

## 03.3 频次函数



## 03.4 替换函数



## 03.5 排序函数

> * 主要包括 `sort_values()` 和 `sort_index()`。这些函数允许根据数据的值或索引进行排序，支持单列、多列、多级索引等不同的排序需求

###  03.5.1 `sort_values()`按值排序

> * 用于按 `DataFrame` 或 `Series` 的值进行排序。可以根据一列或多列的数据来排序

```python
import pandas as pd

# 创建示例 DataFrame
df = pd.DataFrame({
    'A': [2, 1, 4, 3],
    'B': [5, 3, 6, 4],
    'C': [9, 7, 2, 1]
})

# 按列 'A' 进行升序排序
sorted_df = df.sort_values(by='A')
print(sorted_df)

# 结果
   A  B  C
1  1  3  7
0  2  5  9
3  3  4  1
2  4  6  2
```

> * 可以传递一个列列表来按多列排序，`pandas` 会首先按第一个列排序，如果第一个列有相同的值，再按第二个列排序，依此类推
>
> * 参数说明：
>
>   **`by`**: 要排序的列或索引标签。可以是单个标签，也可以是标签列表
>
>   **`ascending`**: 指定排序顺序，默认为 `True`（升序）。可以传递一个布尔值列表，控制每个列的排序顺序
>
>   **`inplace`**: 如果设置为 `True`，则在原地排序，不返回新对象。默认为 `False`
>
>   **`na_position`**: `first` 或 `last`，指定缺失值放在排序后的第一位还是最后一位
>
>   **`ignore_index`**: 如果设置为 `True`，则返回结果会使用默认索引而不是原始数据的索引

```python
# 按列 'A' 和 'B' 进行排序，先按 'A' 升序，再按 'B' 降序
sorted_df = df.sort_values(by=['A', 'B'], ascending=[True, False])
print(sorted_df)

# 结果
   A  B  C
1  1  3  7
0  2  5  9
3  3  4  1
2  4  6  2
```

### 03.5.2 `sort_index()`按索引排序

> * 用于按索引（行标签或列标签）进行排序，适用于 `Series` 和 `DataFrame`

```python
# 按行索引（index）排序
sorted_df = df.sort_index()
print(sorted_df)

# 按列索引排序
sorted_df = df.sort_index(axis=1)
print(sorted_df)
```

> * 对于具有多级索引的数据，你可以通过指定 `level` 参数来排序特定级别的索引
>
> * #### 参数详解
>
>   - **`axis`**: 指定排序方向。`0` 为按行索引排序，`1` 为按列索引排序
>   - **`level`**: 对多级索引数据，可以指定需要排序的级别。可以是单个级别名称或级别列表
>   - **`ascending`**: 同样可以指定排序顺序
>   - **`inplace`**: 是否原地排序
>   - **`na_position`**: 缺失值的位置，同 `sort_values()`

```python
# 创建一个具有多级索引的 DataFrame
import pandas as pd
arrays = [['a', 'a', 'b', 'b'], [2, 1, 2, 1]]
index = pd.MultiIndex.from_arrays(arrays, names=('letter', 'number'))
df_multi = pd.DataFrame({'A': [1, 2, 3, 4]}, index=index)
df_multi
```

```python
# 结果
			A
letter	number	
a		2	1
		1	2
b		2	3
		1	4
```

```python
# 按第二级索引 'number' 进行排序
sorted_df_multi = df_multi.sort_index(level=1)
print(sorted_df_multi)
```

```python
# 结果
			   A
letter number   
a      1       2
b      1       4
a      2       1
b      2       3
```

# 04 窗口











