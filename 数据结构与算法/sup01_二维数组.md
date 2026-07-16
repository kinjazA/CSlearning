# Python 二维数组（Matrix）学习笔记

> 🎯 **目标读者**：正在学习图论 / 邻接矩阵，需要补齐二维数组基础的同学

---

## 一、二维数组的本质

在 Python 中，**没有内置的二维数组类型**。我们所说的"二维数组"，本质上是一个 **列表的列表（list of lists）**，外层列表的每一个元素，本身也是一个列表

```python
# 一个 3 行 4 列的二维数组
matrix: list[list[int]] = [
    [1,  2,  3,  4],   # 第 0 行（i = 0）
    [5,  6,  7,  8],   # 第 1 行（i = 1）
    [9, 10, 11, 12],   # 第 2 行（i = 2）
]
```

### `matrix[i][j]` 到底在访问什么？

把它拆成两步来理解：

```python
row: list[int] = matrix[i]   # 第 1 步：取出第 i 行（一个列表）
element: int   = row[j]      # 第 2 步：从该行中取出第 j 个元素
```

| 符号 | 含义 | 类比 |
|------|------|------|
| `i` | **行索引**（row index） | 你在第几层楼 |
| `j` | **列索引**（column index） | 你在这层楼的第几个房间 |

> 💡 **助记**：先行后列——先选"哪一行"，再选"哪一列"。与数学中矩阵元素 $a_{ij}$的下标顺序完全一致

---

## 二、创建二维数组的陷阱（⚠️ 重点）

### 2.1 ✅ 正确方式：列表推导式

```python
def create_matrix(m: int, n: int, fill: int = 0) -> list[list[int]]:
    """
    创建一个 m 行 n 列、初始值为 fill 的二维数组。
    
    列表推导式语法回顾：
        [expression for variable in iterable]
        含义：对 iterable 中的每个 variable，计算 expression，收集为新列表。
    """
    # 外层循环跑 m 次 → 生成 m 个「独立的」行
    # 内层 [fill] * n → 生成一个长度为 n 的列表
    return [[fill] * n for _ in range(m)]
    #                  ^^^
    #  _ 是 Python 惯例：表示"我不需要这个循环变量"


# 示例：创建 3×4 全 0 矩阵
matrix: list[list[int]] = create_matrix(3, 4)
print(matrix)
# 输出: [[0, 0, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 0]]
```

**关键点**：列表推导式每次迭代都会执行 `[fill] * n`，因此每一行都是一个**全新的、独立的**列表对象

### 2.2 ❌ 错误方式：乘法复制

```python
# ⚠️ 这是一个经典陷阱！
wrong_matrix: list[list[int]] = [[0] * 4] * 3
print(wrong_matrix)
# 看起来一切正常: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

表面上没有问题，但当你**修改其中一个元素**时：

```python
wrong_matrix[0][0] = 99   # 只想改第 0 行第 0 列
print(wrong_matrix)
# 💥 输出: [[99, 0, 0, 0],
#           [99, 0, 0, 0],   ← 第 1 行也被改了！
#           [99, 0, 0, 0]]   ← 第 2 行也被改了！
```

### 2.3 为什么会这样？——引用传递深度解析

根本原因在于 Python 中**列表是引用类型（mutable object）**

```
[[0] * 4] * 3  的执行过程：
```

```
步骤 1：[0] * 4  →  创建一个列表对象，假设内存地址为 0xABC
         0xABC → [0, 0, 0, 0]

步骤 2：[该对象] * 3  →  把 0xABC 这个"地址"复制了 3 份

         wrong_matrix[0] ──→ 0xABC → [0, 0, 0, 0]
         wrong_matrix[1] ──→ 0xABC → 同一个对象！
         wrong_matrix[2] ──→ 0xABC → 同一个对象！
```

**三个"行"指向的是同一个列表对象**。修改任何一个，其余全部同步变化。而列表推导式的过程则是：

```
[  [0]*4  for _ in range(3)  ]  的执行过程：

第 0 次迭代：[0]*4 → 新建 0xAAA → [0, 0, 0, 0]
第 1 次迭代：[0]*4 → 新建 0xBBB → [0, 0, 0, 0]
第 2 次迭代：[0]*4 → 新建 0xCCC → [0, 0, 0, 0]

         matrix[0] ──→ 0xAAA   ← 独立对象
         matrix[1] ──→ 0xBBB   ← 独立对象
         matrix[2] ──→ 0xCCC   ← 独立对象
```

> ⚠️ **铁律**：凡是创建"二维可变结构"，**永远用列表推导式**，不要用 `* m` 复制

---

## 三、核心操作

### 3.1 获取行数和列数

```python
matrix: list[list[int]] = [
    [1, 2, 3],
    [4, 5, 6],
]

rows: int = len(matrix)        # 行数 = 外层列表的长度 → 2
cols: int = len(matrix[0])     # 列数 = 第一行的长度   → 3

# len() 是 Python 内置函数，返回容器中元素的个数

print(f"矩阵大小：{rows} × {cols}")  # 输出：矩阵大小：2 × 3
```

> 💡 `len(matrix[0])` 隐含了一个前提：**矩阵不为空且每行等长**。在实际代码中，如果可能出现空矩阵，请先做防御性检查

### 3.2 双重循环遍历

**方式一：使用索引遍历（最常用，图论中必备）**

```python
def print_matrix(matrix: list[list[int]]) -> None:
    """按行列索引遍历并打印矩阵每个元素。"""
    rows: int = len(matrix)
    cols: int = len(matrix[0])
    
    # range(n) 生成 0, 1, 2, ..., n-1 的序列
    for i in range(rows):        # i 遍历每一行
        for j in range(cols):    # j 遍历当前行的每一列
            print(f"matrix[{i}][{j}] = {matrix[i][j]}")
```

**方式二：使用 `enumerate` 遍历（同时需要索引和值时更 Pythonic）**

```python
def print_matrix_v2(matrix: list[list[int]]) -> None:
    """
    使用 enumerate 遍历。
    
    enumerate() 语法回顾：
        enumerate(iterable) → 生成 (index, element) 的元组序列
        例如 enumerate(['a','b']) → (0,'a'), (1,'b')
    """
    for i, row in enumerate(matrix):      # i = 行号, row = 当前行（列表）
        for j, val in enumerate(row):     # j = 列号, val = 当前元素
            print(f"[{i}][{j}] = {val}")
```

**方式三：仅需要元素值时的简洁写法**

```python
# 当你不需要索引，只关心值本身
for row in matrix:
    for val in row:
        print(val, end=" ")
    print()  # 每行结束后换行
```

---

## 四、二维数组与图论的桥梁——邻接矩阵

### 4.1 什么是邻接矩阵？

对一个有$ V$个顶点的图，邻接矩阵是一个$ V \times V$的数组。其中 `matrix[i][j]` 表示顶点$ i$到顶点 $j$之间**是否有边**（或边的**权重**）

### 4.2 用二维数组表示一个 4 顶点无向图

假设有 4 个顶点（编号 0、1、2、3），初始时没有任何边：

```python
V: int = 4  # 顶点数量

# 创建 V×V 的邻接矩阵，初始值 0 表示"无边"
# 注意：这里必须用列表推导式！
adj_matrix: list[list[int]] = [[0] * V for _ in range(V)]

# 此时的邻接矩阵：
# 顶点   0  1  2  3
#   0  [ 0, 0, 0, 0 ]
#   1  [ 0, 0, 0, 0 ]
#   2  [ 0, 0, 0, 0 ]
#   3  [ 0, 0, 0, 0 ]
```

### 4.3 添加/更新边

**需求**：顶点 1 到顶点 3 之间有一条权重为 5 的边。

```python
def add_edge(adj: list[list[int]], u: int, v: int, weight: int = 1) -> None:
    """
    在无向图的邻接矩阵中添加一条边。
    
    参数:
        adj:    邻接矩阵
        u, v:   边的两个端点
        weight: 边的权重（默认为 1，即无权图）
    """
    adj[u][v] = weight
    adj[v][u] = weight   # 无向图：对称！u→v 和 v→u 都要设置
    # ⚠️ 如果是有向图，只需要设置 adj[u][v] = weight


# 添加边：顶点1 ↔ 顶点3，权重 5
add_edge(adj_matrix, u=1, v=3, weight=5)

# 再添加几条边
add_edge(adj_matrix, u=0, v=1, weight=2)
add_edge(adj_matrix, u=0, v=2, weight=3)
add_edge(adj_matrix, u=2, v=3, weight=1)
```

此时邻接矩阵变为：

```
顶点    0   1   2   3
  0  [  0,  2,  3,  0 ]    ← 顶点 0 与 1（权重2）、2（权重3）相连
  1  [  2,  0,  0,  5 ]    ← 顶点 1 与 0（权重2）、3（权重5）相连
  2  [  3,  0,  0,  1 ]    ← 顶点 2 与 0（权重3）、3（权重1）相连
  3  [  0,  5,  1,  0 ]    ← 顶点 3 与 1（权重5）、2（权重1）相连
```

> 💡 **观察**：无向图的邻接矩阵一定是**关于主对角线对称**的，即 `adj[i][j] == adj[j][i]`

### 4.4 查询：两个顶点之间是否有边？

```python
def has_edge(adj: list[list[int]], u: int, v: int) -> bool:
    """检查顶点 u 和顶点 v 之间是否存在边。"""
    return adj[u][v] != 0


print(has_edge(adj_matrix, 1, 3))  # True  — 权重为 5
print(has_edge(adj_matrix, 0, 3))  # False — 没有直接相连
```

### 4.5 查询：某个顶点的所有邻居

```python
def get_neighbors(adj: list[list[int]], u: int) -> list[int]:
    """
    返回顶点 u 的所有邻居顶点编号。
    
    语法回顾 — 带条件的列表推导式：
        [expression for var in iterable if condition]
        只有满足 condition 的元素才会被收集。
    """
    return [v for v in range(len(adj[u])) if adj[u][v] != 0]


print(get_neighbors(adj_matrix, 0))  # [1, 2] — 顶点 0 的邻居
print(get_neighbors(adj_matrix, 3))  # [1, 2] — 顶点 3 的邻居
```

### 4.6 增加节点

```python
def add_vertex(val: int):
    """添加顶点"""
    cols: int = len(matrix[0]) 
    # 向顶点列表中添加新顶点的值
    vertices.append(val)
    # 在矩阵中添加一行
    new_row = [0] * cols
    matrix.append(new_row)
    # 在邻接矩阵中添加一列
    for row in matrix: # 这里是把每一行取出来，然后每一行末尾append一个0，从而遍历全部行，实现添加一列的效果
        row.append(0)
```

---

## 五、小练习

### 题目：计算矩阵每一行的和

给定一个二维数组，返回一个列表，其中第 \(i\) 个元素是矩阵第 \(i\) 行所有元素的和。

```
输入：
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

输出：[6, 15, 24]
解释：第 0 行 1+2+3=6，第 1 行 4+5+6=15，第 2 行 7+8+9=24
```

### 参考实现

**方法一：基础双重循环**

```python
def row_sums_basic(matrix: list[list[int]]) -> list[int]:
    """用显式循环计算每一行的和。"""
    result: list[int] = []
    for row in matrix:
        total: int = 0
        for val in row:
            total += val
        result.append(total)
        # append() 在列表末尾添加一个元素
    return result
```

**方法二：使用内置函数 `sum()`（推荐）**

```python
def row_sums_builtin(matrix: list[list[int]]) -> list[int]:
    """
    用 sum() + 列表推导式，一行搞定。
    
    sum(iterable) 语法回顾：
        返回可迭代对象中所有元素的总和。
        例如 sum([1, 2, 3]) → 6
    """
    return [sum(row) for row in matrix]
```

**验证**

```python
matrix: list[list[int]] = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

print(row_sums_basic(matrix))    # [6, 15, 24]
print(row_sums_builtin(matrix))  # [6, 15, 24]
```

---

## 附：速查对照表

| 操作 | 代码 | 时间复杂度 |
|------|------|-----------|
| 创建 m×n 零矩阵 | `[[0]*n for _ in range(m)]` | $O(mn)$ |
| 访问元素 | `matrix[i][j]` | $O(1)$ |
| 获取行数 | `len(matrix)` | $O(1)$ |
| 获取列数 | `len(matrix[0])` | $O(1)$ |
| 遍历全部元素 | 双重 `for` 循环 | $O(mn)$ |
| 添加无向边 | `adj[u][v] = adj[v][u] = w` | $O(1)$ |
| 查询是否有边 | `adj[u][v] != 0` | $O(1)$ |
| 查询邻居 | 遍历 `adj[u]` | $O(V)$ |

> 📌 掌握了这份笔记的内容，你就具备了用邻接矩阵实现 BFS、DFS 等图算法的基础
