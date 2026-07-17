# 01 单级索引

> * 顾名思义，单级索引就是用一列指标区分样本，比方说每个身份证对应一个人，就是单级索引的意思

## 01.1 `DataFrame`的列索引

> * 列索引（Column Indexing）是指如何访问、选择和操作 `DataFrame` 中的列

### 01.1.1 单列、多列索引

> * ==通过[列名]可以从`DataFrame`中取出相应的列，返回值为`Series`==
> * ==可以通过[列名组成的列表]的方式来得到多列，其返回值为一个`DataFrame`==

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9],
}

df = pd.DataFrame(data)

# 选择单列 'A'
col_A = df['A']
print(col_A)

# 选择列 'A' 和 'B'
cols_AB = df[['A', 'B']]
print(cols_AB)
```

### 01.1.2 属性方式索引

> * 对于列名是合法的 Python 标识符（即列名不包含空格或特殊字符，并且不以数字开头），你可以使用点符号（`.`）访问列。这种方式更加简洁，但是对于复杂列名并不适用

```python
# 选择列 'A'
col_A = df.A
print(col_A)

```

## 01.2 `Series`的行索引

> * 行索引（Row Indexing）指的是如何访问、选择、操作 `Series` 中的元素
> * 考虑到数据分析时，主要是`DataFrame`，这部分先略过

## 01.3 `loc`索引器

> * `loc` 索引器是一个基于标签的访问器，常用于 `DataFrame` 和 `Series` 数据结构中。通过 `loc` 索引器，可以使用行索引标签、列标签或它们的组合来访问、切片和修改数据
> * `row_indexer`: 行索引标签或标签切片，用于选择 `DataFrame` 中的行
> * `column_indexer`: 列索引标签或标签切片，用于选择 `DataFrame` 中的列

```python
# 基本语法
DataFrame.loc[row_indexer, column_indexer]
```

### 01.3.1 选择行

> * ==标签切片允许选择一系列行，`loc` 会包括切片的终止标签==

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9],
}

df = pd.DataFrame(data, index=['x', 'y', 'z'])

# 通过标签 'y' 选择行
row_y = df.loc['y']
print(row_y)

# 通过标签列表选择行 'x' 和 'z'
rows_xz = df.loc[['x', 'z']]
print(rows_xz)

# 通过标签切片选择行 'x' 到 'y'
rows_xy = df.loc['x':'y']
print(rows_xy)

```

### 01.3.2  选择列

```python
# 通过标签 'A' 选择列
col_A = df.loc[:, 'A']
print(col_A)

# 通过标签列表选择列 'A' 和 'C'
cols_AC = df.loc[:, ['A', 'C']]
print(cols_AC)

# 通过标签切片选择从 'A' 到 'B' 的列
cols_AB = df.loc[:, 'A':'B']
print(cols_AB)

```

### 01.3.3 选择行和列

```python
# 选择 'y' 行和 'B' 列的单个值
val_y_B = df.loc['y', 'B']
print(val_y_B)

# 选择 'y' 行和 'A', 'C' 列的值
vals_y_AC = df.loc['y', ['A', 'C']]
print(vals_y_AC)

# 选择 'x', 'z' 行和 'A', 'B' 列的值
subset_xz_AB = df.loc[['x', 'z'], ['A', 'B']]
print(subset_xz_AB)

```

### 01.3.4 使用布尔索引进行条件选择

> * `loc`还可以结合布尔表达式，用于根据条件筛选数据

```python
# 选择列 'A' 中值大于 1 的所有行
filtered_rows = df.loc[df['A'] > 1]
print(filtered_rows)

# 选择列 'A' 中值大于 1 的所有行，且只选 'B' 列
filtered_col_B = df.loc[df['A'] > 1, 'B']
print(filtered_col_B)

```

### 01.3.5 修改数据

> * `loc` 也可以用于修改 `DataFrame` 中的数据

```python
# 修改 'z' 行 'C' 列的值为 99
df.loc['z', 'C'] = 99
print(df)

# 修改 'A' 列的所有值为 0
df.loc[:, 'A'] = 0
print(df)

# 将 'B' 列中大于 4 的值修改为 100
df.loc[df['B'] > 4, 'B'] = 100
print(df)

```

## 01.4 `iloc`索引器

> * `iloc` 索引器是一种基于整数位置的访问器，用于 `DataFrame` 和 `Series` 中。通过 `iloc`，可以使用整数位置索引来访问、切片和修改数据。这与 `loc` 索引器不同，`loc` 是基于标签的，而 `iloc` 是基于位置的
> * `row_indexer`: 用于选择行的整数位置索引或索引切片
> * `column_indexer`: 用于选择列的整数位置索引或索引切片

```python
DataFrame.iloc[row_indexer, column_indexer]
```

### 01.4.1 选择行

> * ==整数切片选择时，切片的终止位置不包括在内，与 Python 的标准切片行为一致==

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9],
}

df = pd.DataFrame(data)

# 通过位置 1 选择第二行
row_1 = df.iloc[1]
print(row_1)

# 通过位置列表选择第一行和第三行
rows_0_2 = df.iloc[[0, 2]]
print(rows_0_2)

# 通过位置切片选择第一行和第二行
rows_0_2 = df.iloc[0:2]
print(rows_0_2)

```

### 01.4.2 选择列

```python
# 通过位置 0 选择第一列
col_0 = df.iloc[:, 0]
print(col_0)

# 通过位置列表选择第一列和第三列
cols_0_2 = df.iloc[:, [0, 2]]
print(cols_0_2)

# 通过位置切片选择第一列和第二列
cols_0_2 = df.iloc[:, 0:2]
print(cols_0_2)

```

### 01.4.3 选择行和列

```python
# 选择第二行和第二列的单个值
val_1_1 = df.iloc[1, 1]
print(val_1_1)

# 选择第二行和第一列、第三列的值
vals_1_0_2 = df.iloc[1, [0, 2]]
print(vals_1_0_2)

# 选择第一行和第三行，以及第一列和第三列
subset_0_2_0_2 = df.iloc[[0, 2], [0, 2]]
print(subset_0_2_0_2)

```

### 01.4.4 使用布尔索引进行条件选择

> * 虽然 `iloc` 主要用于整数位置索引，但也可以结合布尔表达式进行筛选，不过布尔表达式通常是基于位置的条件

```python
# 使用布尔数组选择行（例如选择前两行）
bool_indexer = [True, True, False]
filtered_rows = df.iloc[bool_indexer]
print(filtered_rows)
```

### 01.4.5 修改数据

```python
# 修改第二行第二列的值为 99
df.iloc[1, 1] = 99
print(df)

# 修改第二列的所有值为 0
df.iloc[:, 1] = 0
print(df)

# 修改第二行的所有值为 -1
df.iloc[1, :] = -1
print(df)

```

> `iloc` 与 `loc` 的区别：
>
> - **基于位置 (`iloc`) vs. 基于标签 (`loc`)**: `iloc` 是通过整数位置索引访问数据， `loc` 是通过标签索引访问数据
> - **切片行为**: `iloc` 的切片与 Python 标准切片一致，不包括终止位置，而 `loc` 的切片包括终止标签

## 01.5 `query()`函数

> * 是一种方便的方式来对 `DataFrame` 进行条件筛选。它允许用户通过类似 SQL 的语法在数据帧中查询数据，而无需使用传统的布尔索引或复杂的表达式。`query()` 函数可以使代码更加简洁易读
> * `expr`: 字符串形式的查询表达式。表达式中可以使用列名来指定筛选条件
> * `inplace`: 是否在原地修改数据帧。默认为 `False`，即返回一个新的 `DataFrame``
> * ``kwargs`: 其他可选参数，比如局部变量的传递

```python
# 基本语法
DataFrame.query(expr, inplace=False, **kwargs)
```

### 01.5.1 简单条件筛选

```python
import pandas as pd

data = {
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': [100, 200, 300, 400]
}

df = pd.DataFrame(data)

# 使用 query() 筛选出 A 列中值大于 2 的行
result = df.query('A > 2')
print(result)
```

### 01.5.2 多个条件筛选

> * 在 `query()` 中，可以使用 `and` 和 `or` 组合多个条件，类似于 Python 的逻辑运算符

```python
# 使用 query() 筛选出 A 列中值大于 2 且 B 列中值小于 40 的行
result = df.query('A > 2 and B < 40')
print(result)
```

### 01.5.3 使用局部变量

> * 查询表达式中需要引用 Python 中的局部变量，可以使用 `@` 符号

```python
threshold = 2

# 使用 query() 筛选出 A 列中值大于 threshold 的行
result = df.query('A > @threshold')
print(result)
```

### 01.5.4 引用包含空格或特殊字符的列名

> * 如果列名包含空格或特殊字符，可以使用反引号 (`) 来引用列名

```python
data = {
    'A column': [1, 2, 3, 4],
    'B column': [10, 20, 30, 40]
}

df = pd.DataFrame(data)

# 使用 query() 筛选出 'A column' 列中值大于 2 的行
result = df.query('`A column` > 2')
print(result)
```

### 01.5.5 复杂查询表达式

```python
threshold_A = 2
threshold_B = 30

# 筛选 A 列大于 2 且 B 列减去 10 的值大于 20 的行
result = df.query('A > @threshold_A and (B - 10) > @threshold_B')
print(result)
```

### 01.5.6 使用 `inplace` 修改原数据

> * 如果想直接在原数据帧中修改，可以将 `inplace` 参数设置为 `True`

```python
# 原地修改 df，将 A 列中大于 2 的行保留
df.query('A > 2', inplace=True)
print(df)
```

### 01.5.7 应用场景

```python
# 示例数据
data = {
    'Stock': ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA'],
    'Date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01'],
    'Price_Open': [150, 1800, 3200, 250, 700],
    'Price_Close': [155, 1750, 3250, 245, 720]
}

df = pd.DataFrame(data)

# 计算涨跌幅度
df['Change_Percent'] = ((df['Price_Close'] - df['Price_Open']) / df['Price_Open']) * 100

# 筛选出涨跌幅度大于 2% 的股票
result = df.query('Change_Percent > 2')
print(result)
```

```python
# 示例数据
data = {
    'Event': ['Event1', 'Event2', 'Event3', 'Event4'],
    'Date': ['2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01'],
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

# 筛选出 2024-08-01 到 2024-09-30 之间发生的事件
result = df.query('"2024-08-01" <= Date <= "2024-09-30"')
print(result)
```

### 01.5.8 索引运算

> * 索引（`Index`）运算允许对不同数据集的索引进行集合运算，例如并集、交集、差集和对称差。通过这些运算，可以高效地比较和合并不同 `DataFrame` 或 `Series` 的索引，从而在数据分析中进行更复杂的操作
> * 通常我们会先将两个数据结构的索引先进行去重（使用.unique()），再进行之后的索引运算

```python
import pandas as pd

# 示例索引
index1 = pd.Index([1, 2, 3, 4, 4, 4])  # 重复的4
index2 = pd.Index([3, 4, 5, 6])

# 手动去重
index1_unique = pd.Index(index1.unique())
index2_unique = pd.Index(index2.unique())
```

> * `union()` 方法用于返回两个索引集合的并集。即返回两个索引集合中所有唯一的索引

具有实际应用场景的用法在后面，这里先空着

# 02 多级索引

> * 感觉我的常用实际场景都是加载一个数据集之后再进行各种处理，所以这里不写太多其他的

## 02.1 设置多级索引

> * 第一个场景就是直接现成的数据，直接指定索引
> * `index_col=[0, 1]`: 指定 `0` 和 `1` 列为索引，会创建一个多级索引（MultiIndex）

```python
import pandas as pd

# 读取 CSV 文件，并将第一列和第二列设置为多级索引
df = pd.read_csv('data.csv', index_col=[0, 1])
```

> * 第二个场景就是先经过处理之后，再把指定的列设置为索引

```python
>>> df = pd.DataFrame({'month': [1, 4, 7, 10],
...                    'year': [2012, 2014, 2013, 2014],
...                    'sale': [55, 40, 84, 31]})
>>> df
   month  year  sale
0      1  2012    55
1      4  2014    40
2      7  2013    84
3     10  2014    31

# 使用列“年”和“月”创建多重索引
>>> df.set_index(['year', 'month'])
            sale
year  month
2012  1     55
2014  4     40
2013  7     84
2014  10    31

# 使用索引和列创建多重索引
>>> df.set_index([pd.Index([1, 2, 3, 4]), 'year'])
         month  sale
   year
1  2012  1      55
2  2014  4      40
3  2013  7      84
4  2014  10     31
```

## 02.2 多级索引中的`loc`索引器

> * ==由于多级索引中的单个元素以元组为单位，因此之前的loc索引器的使用方法和iloc索引器的使用方法完全可以照搬，只需把标量的位置替换成对应的元组==

```python
# 数据样式
           Value1  Value2
Upper Lower               
A     one      10     100
      two      20     200
B     one      30     300
      two      40     400
```

```python
# 访问 A 组的数据
print(df.loc['A'])

# 结果
       Value1  Value2
Lower                 
one        10     100
two        20     200
```

```python
# 访问 A 组中 'one' 层的数据
print(df.loc[('A', 'one')])

# 结果
Value1     10
Value2    100
Name: (A, one), dtype: int64
```

# 03 常用索引方法























