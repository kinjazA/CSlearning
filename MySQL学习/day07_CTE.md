# 01 CTE（WITH 语句）

CTE（Common Table Expression，公共表表达式）本质上是一个**命名的临时结果集**，只在当前查询的生命周期内存在。它不是物理表，也不是视图，可以理解为"给一段子查询起个名字，然后反复引用"

| 对比维度 | 子查询             | CTE                |
| -------- | ------------------ | ------------------ |
| 可读性   | 嵌套深，难以阅读   | 逻辑分层，清晰     |
| 复用性   | 同一逻辑需重复写   | 定义一次，多次引用 |
| 调试性   | 难以单独测试某层   | 可以逐段注释调试   |
| 执行方式 | 每次引用都重新执行 | 视优化器决定       |

```sql
-- 基本语法
WITH cte名称 AS (
    -- 这里写子查询
    SELECT ...
)
SELECT *
FROM cte名称
WHERE ...;
```



```sql
-- 找出每个部门中，薪资高于本部门平均薪资的员工
-- 第一步：先用 CTE 计算每个部门的平均薪资
WITH dept_avg AS (
    SELECT
        dept_id,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY dept_id
)
-- 第二步：主查询中关联使用
SELECT
    e.name,
    e.dept_id,
    e.salary,
    d.avg_salary
FROM employees e
JOIN dept_avg d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_salary;
```

`WITH` 块在逻辑上"先执行"，主查询再引用它。实际上 MySQL 优化器可能将其内联（inline）处理，也可能物化（materialize）为临时表，取决于查询复杂度

# 02 多个 CTE（嵌套/链式）

多个 CTE 之间用逗号分隔，**后面的 CTE 可以引用前面的 CTE**，这是最强大的特性：

```sql
WITH
-- 第一层：计算每日销售额
daily_sales AS (
    SELECT
        DATE(order_time) AS sale_date,
        SUM(amount)      AS daily_amount
    FROM orders
    GROUP BY DATE(order_time)
),
-- 第二层：在第一层基础上，计算7日滑动平均
rolling_avg AS (
    SELECT
        sale_date,
        daily_amount,
        AVG(daily_amount) OVER (
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS avg_7day
    FROM daily_sales  -- 引用了上面的 CTE
)
-- 主查询
SELECT *
FROM rolling_avg
WHERE sale_date >= '2024-01-01';
```

**数据分析常用模式：** 第一层做数据清洗/聚合 → 第二层做指标计算 → 第三层做最终筛选/排名，层层递进，逻辑清晰



递归 CTE 是 CTE 的特殊形式，允许查询**引用自身**，用来处理两类经典问题：

1. **层级/树形结构**（组织架构、类目树）
2. **序列生成**（连续日期、数字序列）

递归 CTE 的结构固定为**三段式**：

```sql
WITH RECURSIVE cte名称 AS (
    -- ① 锚点成员（Anchor Member）：初始数据，只执行一次
    SELECT ...

    UNION ALL

    -- ② 递归成员（Recursive Member）：引用自身，反复执行直到无新数据
    SELECT ...
    FROM cte名称          -- 引用自身！
    WHERE ...             -- 终止条件（必须有，否则死循环）
)
SELECT * FROM cte名称;
```

**执行逻辑图：**

```css
第1次：执行锚点 → 结果集 R1
第2次：将 R1 代入递归成员 → 结果集 R2
第3次：将 R2 代入递归成员 → 结果集 R3
...
直到某次执行返回空集 → 停止
最终结果 = R1 ∪ R2 ∪ R3 ∪ ...
```



 场景一：层级查询（组织架构树）

```css
-- employees 表
-- id | name   | manager_id
-- 1  | CEO    | NULL
-- 2  | CTO    | 1
-- 3  | CFO    | 1
-- 4  | Dev_A  | 2
-- 5  | Dev_B  | 2
```

**目标：** 从 CEO 出发，列出所有汇报链路及层级深度

```sql
WITH RECURSIVE org_tree AS (
    -- ① 锚点：从根节点（CEO）开始
    SELECT
        id,
        name,
        manager_id,
        0        AS depth,       -- 层级深度
        name     AS path         -- 路径
    FROM employees
    WHERE manager_id IS NULL     -- 根节点条件

    UNION ALL

    -- ② 递归：找到当前层级的所有下属
    SELECT
        e.id,
        e.name,
        e.manager_id,
        ot.depth + 1,                           -- 深度 +1
        CONCAT(ot.path, ' → ', e.name)          -- 拼接路径
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id    -- 关键：子节点的 manager_id = 父节点的 id
)
SELECT
    LPAD('', depth * 4, ' ') AS indent,   -- 用空格展示缩进
    name,
    depth,
    path
FROM org_tree
ORDER BY path;
```

------

场景二：生成连续日期序列

这在数据分析中**极其常用**——生成日期维度表，用来 LEFT JOIN 业务数据，避免日期断档导致数据缺失

```sql
WITH RECURSIVE date_series AS (
    -- ① 锚点：起始日期
    SELECT DATE('2024-01-01') AS dt

    UNION ALL

    -- ② 递归：每次加一天
    SELECT DATE_ADD(dt, INTERVAL 1 DAY)
    FROM date_series
    WHERE dt < '2024-01-31'   -- 终止条件：不超过结束日期
)
SELECT dt FROM date_series;
-- 输出：2024-01-01, 2024-01-02, ..., 2024-01-31
```

**实战用法：** 与销售数据 LEFT JOIN，补全没有订单的日期：

```sql
WITH RECURSIVE date_series AS (
    SELECT DATE('2024-01-01') AS dt
    UNION ALL
    SELECT DATE_ADD(dt, INTERVAL 1 DAY) FROM date_series WHERE dt < '2024-01-31'
)
SELECT
    ds.dt,
    COALESCE(SUM(o.amount), 0) AS daily_amount   -- 无订单的日期补0
FROM date_series ds
LEFT JOIN orders o ON DATE(o.order_time) = ds.dt
GROUP BY ds.dt
ORDER BY ds.dt;
```

> ⚠️ **注意：** MySQL 默认递归深度上限为 1000，可通过 `SET SESSION cte_max_recursion_depth = 10000;` 调整



# 03 EXPLAIN 执行计划

在数据分析工作中，会频繁地写复杂的 SQL。一条 SQL 写出来能跑出结果，只是最低要求。更高的要求是：**这条 SQL 在大数据量下也能快速跑完**，而不是跑几分钟甚至几十分钟

`EXPLAIN` 就是 MySQL 提供的"透视眼"——它让你在不实际执行 SQL 的情况下，看清楚 MySQL 打算用什么方式来执行这条 SQL，从而判断是否需要优化。`EXPLAIN` 的用法非常简单，在任何 `SELECT` 语句前面加上 `EXPLAIN` 关键字即可：

```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 100;
```

从 MySQL 8.0 开始，还可以使用 `EXPLAIN ANALYZE`，它不仅展示执行计划，还真实执行 SQL 并展示**实际耗时**和**实际行数**，对比"预估"和"实际"的差异，是更强大的诊断工具：

```sql
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 100;
```

