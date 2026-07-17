# 01 Ndarray对象

numpy主要用于数学、科学计算、模型运算

> * NumPy 最重要的一个特点是其 N 维数组对象 ndarray，它是一系列同类型数据的集合，以 0 下标为开始进行集合中元素的索引
> * ndarray 对象是用于存放同类型元素的多维数组
> * ndarray 中的每个元素在内存中都有相同存储大小的区域

```python
import numpy as np 
# numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
a = np.array([1,2,3])  
print (a)
# [1 2 3]

# 多于一个维度  
a = np.array([[1,  2],  [3,  4]])  
print (a)
# [[1  2] 
# [3  4]]
```

# 02 Numpy属性

> * NumPy 数组的维数称为秩（rank），秩就是轴的数量，即数组的维度，一维数组的秩为 1，二维数组的秩为 2，以此类推
> * axis=0，表示沿着第 0 轴进行操作，即对每一列进行操作；axis=1，表示沿着第1轴进行操作，即对每一行进行操作

![image-20251201214934222](F:\note\DataAnalysis\数据分析\numpy笔记\day01_创建数组.assets\image-20251201214934222.png)

```python
import numpy as np 
 
 # 查看数组维度
a = np.arange(24)  
print (a.ndim)             # a 现只有一个维度
# 现在调整其大小
b = a.reshape(2,4,3)  # b 现在拥有三个维度
print (b.ndim)

# 查看数组形状
a = np.array([[1,2,3],[4,5,6]])  
print (a.shape)
# (2, 3)

# 调整数组形状
a = np.array([[1,2,3],[4,5,6]]) 
a.shape =  (3,2)  
print (a)
# [[1 2]
# [3 4]
# [5 6]]

# 或者是用reshape
a = np.array([[1,2,3],[4,5,6]]) 
b = a.reshape(3,2)  
print (b)
```

# 03 创建数组

## 03.1 预定义数组形状

> * numpy.empty 方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组：

![image-20251201220332124](F:\note\python_learning\数据分析\numpy笔记\day01_创建数组.assets\image-20251201220332124.png)

```python
import numpy as np 
x = np.empty([3,2], dtype = int) 
print (x)
```

> * numpy.zeros创建指定大小的数组，数组元素以 0 来填充：

```python
# 默认为浮点数
x = np.zeros(5) 
print(x)
 
# 设置类型为整数
y = np.zeros((5,), dtype = int) 
print(y)
 
# 自定义类型
z = np.zeros((2,2), dtype = [('x', 'i4'), ('y', 'i4')])  
print(z)
```

> * numpy.ones创建指定形状的数组，数组元素以 1 来填充：

```python
# 默认为浮点数
x = np.ones(5) 
print(x)
 
# 自定义类型
x = np.ones([2,2], dtype = int)
print(x)
```

> * numpy.zeros_like 用于创建一个与给定数组具有相同形状的数组，数组元素以 0 来填充
> * numpy.zeros 和 numpy.zeros_like 都是用于创建一个指定形状的数组，其所有元素都是 0
> * 它们的区别在于：numpy.zeros 可以直接指定要创建的数组的形状，而 numpy.zeros_like 则是创建一个与给定数组具有相同形状的数组
> * 同样的还有numpy.ones_like函数

```python
# 创建一个 3x3 的二维数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 
# 创建一个与 arr 形状相同的，所有元素都为 0 的数组
zeros_arr = np.zeros_like(arr)
print(zeros_arr)
```

> * numpy.full函数，用来指定生成包含某个指定数字的给定形状的数组

```python
arr = np.full((3,4),2026) # 创建的三行四列，全部
```

## 03.2 从已有数组创建

> * numpy.asarray 类似 numpy.array，但 numpy.asarray 参数只有三个，比 numpy.array 少两个
> * numpy.asarray(a, dtype = None, order = None)

```python
import numpy as np 
 
x =  [1,2,3] 
a = np.asarray(x)  
print (a)
# [1  2  3]

# 将元组转换为 ndarray
x =  (1,2,3) 
a = np.asarray(x)  
print (a)

# 将元组列表转换为 ndarray
x =  [(1,2,3),(4,5)] 
a = np.asarray(x)  
print (a)
```

## 03.3 从数值范围创建

> * numpy.arange(start, stop, step, dtype),左闭右开

![image-20251202150657532](F:\note\python_learning\数据分析\numpy笔记\day01_创建数组.assets\image-20251202150657532.png)

```python
import numpy as np
arr = np.arange(1,10,2)
# [1 3 5 7 9]
```

> * numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的

![image-20251202150914865](F:\note\python_learning\数据分析\numpy笔记\day01_创建数组.assets\image-20251202150914865.png)

```python
import numpy as np
a = np.linspace(1,1,10)
print(a)
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
```

> * numpy.logspace 函数用于创建一个于等比数列
> * np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)

![image-20251202151911116](F:\note\python_learning\数据分析\numpy笔记\day01_创建数组.assets\image-20251202151911116.png)

```python
import numpy as np
a = np.logspace(0,9,10,base=2)
print (a)
# [  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
```

## 03.4 特殊矩阵的创建

```python
# 生成单位矩阵
import numpy as np
arr1 = np.eye(3)  # identity 单位矩阵，与eye谐音
print(arr1)  # 三阶单位矩阵


# 生成对角矩阵
arr2 = np.diag([1,2,3,4])  # 这里是给定对角线元素
print(arr2)
```

```python
# 随机数组的生成（下面三个全是基于均匀分布生成的）
# 生成0~1之间的随机浮点数
arr3 = np.random.rand(2,3)
print(arr3)

# 生成指定范围区间的随机浮点数
arr4 = np.random.uniform(3,6,(2,3))  # 生成3到6之间的2行3列随机矩阵

# 生成指定范围区间的随机整数
arr4 = np.random.randint(3,6,(2,3))  # 生成3到6之间的2行3列随机矩阵
```

```python
下面是基于标准正态分布生成的随机数组
arr5 = np.random.randn(2,3)  # 大致是-3~3之间，符合3西格玛原则
```

```python
# 设置随机种子
np.random.seed(20)  # 设置了之后，后面生成的随机数就可以保持不变
```



















