## 1. 什么是窗口函数

### 1.1 核心概念

窗口函数（Window Function），也叫**分析函数**，是一种对一组相关行（称为"窗口"）进行计算的函数

**与 GROUP BY 的关键区别：**

| 对比项 | GROUP BY 聚合 | 窗口函数 |
|--------|--------------|---------|
| 返回行数 | 压缩为一行（每组） | **保留所有原始行** |
| 是否能同时看到明细和汇总 | 否 | **是** |
| 典型使用场景 | 统计每组总量 | 排名、累计、环比 |

**形象理解：**
想象一个"滑动的窗口"在数据上移动，每次对窗口内的数据做计算，但计算完成后**不改变行数**，只是在原有行上新增一列计算结果

---

## 2. 基本语法结构

```sql
函数名([参数]) OVER (
    [PARTITION BY 分组列]   -- 可选：按哪列分组（类似GROUP BY，但不压缩行）
    [ORDER BY 排序列 ASC|DESC]  -- 可选：组内排序方式
    [ROWS|RANGE BETWEEN ... AND ...]  -- 可选：窗口帧范围
)
```

### 2.1 各子句说明

| 子句 | 是否必须 | 作用 |
|------|---------|------|
| `PARTITION BY` | 否 | 将数据分成多个独立的窗口分区，每个分区内独立计算 |
| `ORDER BY` | 否（排名/偏移函数通常必须） | 定义分区内的排序规则，影响排名和累计计算 |
| `ROWS/RANGE BETWEEN` | 否 | 精确控制每行计算时纳入的行范围（窗口帧） |

窗口函数各子句执行顺序：

```sql
FROM
WHERE
GROUP BY
HAVING
WINDOW FUNCTION
SELECT
ORDER BY
```

---

### 2.2 函数分类总览

```css
窗口函数
├── 排名函数
│   ├── ROW_NUMBER()    -- 连续不重复排名（1,2,3,4）
│   ├── RANK()          -- 跳跃排名（1,1,3,4）
│   ├── DENSE_RANK()    -- 连续排名（1,1,2,3）
│   └── NTILE(n)        -- 分桶，将数据切成n份
│
├── 偏移函数
│   ├── LAG(col, n)     -- 取当前行向上偏移n行的值（上一行）
│   └── LEAD(col, n)    -- 取当前行向下偏移n行的值（下一行）
│
├── 聚合窗口函数
│   ├── SUM()           -- 累计求和
│   ├── AVG()           -- 移动平均
│   ├── COUNT()         -- 累计计数
│   ├── MAX()           -- 窗口内最大值
│   └── MIN()           -- 窗口内最小值
│
└── 分布函数
    ├── PERCENT_RANK()  -- 百分比排名（0~1）
    └── CUME_DIST()     -- 累积分布（0~1）
```

---

## 3. 准备测试数据

> 以下所有示例均基于这套电商数据表

```sql
-- ============================================================
-- 建表：用户表
-- ============================================================
CREATE TABLE users (
    user_id   INT PRIMARY KEY,
    user_name VARCHAR(50),
    city      VARCHAR(50),
    reg_date  DATE           -- 注册日期
);

INSERT INTO users VALUES
(1, '张三', '北京', '2023-01-10'),
(2, '李四', '上海', '2023-02-15'),
(3, '王五', '北京', '2023-03-20'),
(4, '赵六', '上海', '2023-01-05'),
(5, '孙七', '广州', '2023-04-01'),
(6, '周八', '广州', '2023-02-28');

-- ============================================================
-- 建表：订单表
-- ============================================================
CREATE TABLE orders (
    order_id    INT PRIMARY KEY,
    user_id     INT,
    product_id  INT,
    category    VARCHAR(50),    -- 商品类目：电子、服装、食品
    amount      DECIMAL(10,2),  -- 订单金额
    order_date  DATE            -- 下单日期
);

INSERT INTO orders VALUES
(1001, 1, 101, '电子', 3200.00, '2024-01-05'),
(1002, 1, 102, '服装',  450.00, '2024-01-20'),
(1003, 2, 103, '食品',   89.00, '2024-01-08'),
(1004, 2, 101, '电子', 5800.00, '2024-02-10'),
(1005, 3, 104, '服装',  320.00, '2024-01-15'),
(1006, 3, 105, '食品',  156.00, '2024-02-20'),
(1007, 4, 101, '电子', 7200.00, '2024-01-25'),
(1008, 4, 106, '服装',  680.00, '2024-02-05'),
(1009, 5, 107, '食品',  210.00, '2024-02-18'),
(1010, 5, 101, '电子', 4500.00, '2024-03-01'),
(1011, 6, 108, '服装',  890.00, '2024-01-30'),
(1012, 6, 103, '食品',  135.00, '2024-03-05'),
(1013, 1, 101, '电子', 2100.00, '2024-03-10'),
(1014, 2, 109, '服装',  560.00, '2024-03-15'),
(1015, 4, 103, '食品',  178.00, '2024-03-20');

-- ============================================================
-- 建表：月度销售汇总表
-- ============================================================
CREATE TABLE monthly_sales (
    category   VARCHAR(50),
    sale_month VARCHAR(7),   -- 格式：YYYY-MM
    revenue    DECIMAL(12,2) -- 当月营收
);

INSERT INTO monthly_sales VALUES
('电子', '2024-01', 120000),
('电子', '2024-02', 135000),
('电子', '2024-03', 98000),
('服装', '2024-01', 45000),
('服装', '2024-02', 52000),
('服装', '2024-03', 67000),
('食品', '2024-01', 18000),
('食品', '2024-02', 22000),
('食品', '2024-03', 19500);
```

---

## 4. 排名函数

### 4.1 ROW_NUMBER() —— 连续不重复排名

**特点：** 即使值相同，排名也不重复，按顺序依次递增

```sql
-- 业务需求：查询每个用户的订单，并按金额从高到低标上序号
SELECT
    user_id,
    order_id,
    amount,
    -- ROW_NUMBER：不管金额是否相同，每行都有唯一序号
    ROW_NUMBER() OVER (
        PARTITION BY user_id     -- 按用户分组，每个用户内部独立排名
        ORDER BY amount DESC     -- 组内按金额从大到小排
    ) AS rn
FROM orders;
```

**结果示意（user_id=1 的部分）：**

| user_id | order_id | amount | rn |
|---------|----------|--------|----|
| 1 | 1001 | 3200.00 | 1 |
| 1 | 1013 | 2100.00 | 2 |
| 1 | 1002 | 450.00 | 3 |

**经典应用：取每个用户金额最高的那一笔订单（Top-N 问题）**

```sql
-- 用子查询包裹，筛选 rn=1 即可取到每人最大订单
SELECT *
FROM (
    SELECT
        user_id,
        order_id,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY amount DESC
        ) AS rn
    FROM orders
) ranked   -- FROM 后面的子查询必须有别名
WHERE rn = 1;   -- 只取每人排名第一的订单
```

---

### 4.2 RANK() —— 跳跃排名

**特点：** 相同值并列排名，但后续名次会跳过（1, 1, 3, 4...）。

```sql
-- 业务需求：对所有用户按总消费额排名，金额相同的用户并列
SELECT
    user_id,
    SUM(amount) AS total_amount,
    -- RANK：有并列时跳跃（如两人并列第1，下一名为第3）
    RANK() OVER (
        ORDER BY SUM(amount) DESC  -- 全局排名，不分区
    ) AS rnk
FROM orders
GROUP BY user_id;
```

> **注意：** `RANK()` 可以和 `GROUP BY` 一起使用，窗口函数作用在 GROUP BY 之后的结果集上。

如果没有 `PARTITION BY`：

```sql
RANK() OVER(ORDER BY amount DESC)
```

就是 **全表排名**。如果有：

```sql
PARTITION BY user_id
```

就是：**每个 user 单独排名**

---

### 4.3 DENSE_RANK() —— 连续排名

**特点：** 相同值并列排名，后续名次**不跳过**（1, 1, 2, 3...）。

```sql
-- 业务需求：对每个类目内的订单按金额排名，使用连续排名
SELECT
    category,
    order_id,
    amount,
    -- DENSE_RANK：有并列时不跳跃（1,1,2,3...）
    DENSE_RANK() OVER (
        PARTITION BY category
        ORDER BY amount DESC
    ) AS dense_rnk,
    -- 对比展示三种排名函数的区别
    RANK() OVER (
        PARTITION BY category
        ORDER BY amount DESC
    ) AS rnk,
    ROW_NUMBER() OVER (
        PARTITION BY category
        ORDER BY amount DESC
    ) AS rn
FROM orders;
```

**三种排名函数对比（当存在相同值时）：**

| 值 | ROW_NUMBER | RANK | DENSE_RANK |
|----|-----------|------|-----------|
| 100 | 1 | 1 | 1 |
| 100 | 2 | 1 | 1 |
| 90 | 3 | 3 | 2 |
| 80 | 4 | 4 | 3 |

---

### 4.4 NTILE(n) —— 分桶函数

**特点：** 将数据均匀切割成 n 份，返回每行所属的桶编号。

```sql
-- 业务需求：将用户按总消费额分成高/中/低三档（消费分层）
SELECT
    user_id,
    total_amount,
    -- NTILE(3)：把所有用户分成3桶，返回桶编号1/2/3
    NTILE(3) OVER (
        ORDER BY total_amount DESC   -- 按消费额从高到低分桶
    ) AS bucket,
    -- 用 CASE 将桶编号转为业务标签
    CASE
    -- 同一层 SELECT 中，别名 bucket 不能马上在另一个表达式里继续引用
        WHEN NTILE(3) OVER (ORDER BY total_amount DESC) = 1 THEN '高价值用户'
        WHEN NTILE(3) OVER (ORDER BY total_amount DESC) = 2 THEN '中等用户'
        ELSE '低活跃用户'
    END AS user_segment
FROM (
    SELECT user_id, SUM(amount) AS total_amount
    FROM orders
    GROUP BY user_id
) t;
```



![image-20260315084414539](day06_%E7%AA%97%E5%8F%A3%E5%87%BD%E6%95%B0.assets/image-20260315084414539.png)

---

## 5. 偏移函数

偏移函数用于获取**同一分区内**当前行的上/下某行的值，是计算**环比、同比、次序差异**的核心工具。

### 5.1 LAG() —— 取上一行的值

> * 这里的偏移行数是指站在本行往上数的第几行，是只有一行，而不是前几行

```sql
-- 语法：LAG(列名, 偏移行数, 默认值)
-- 偏移行数默认为1，默认值在没有上一行时使用

-- 业务需求：计算每个类目每月的环比增长额和增长率
SELECT
    category,
    sale_month,
    revenue,
    -- LAG：取上一行（即上个月）的营收
    LAG(revenue, 1, 0) OVER (
        PARTITION BY category     -- 按类目分区，每个类目独立计算
        ORDER BY sale_month ASC   -- 按月份升序排列
    ) AS last_month_revenue,
    -- 计算环比增长额 = 本月 - 上月
    revenue - LAG(revenue, 1, 0) OVER (
        PARTITION BY category
        ORDER BY sale_month
    ) AS mom_growth,
    -- 计算环比增长率 = (本月 - 上月) / 上月 * 100%
    ROUND(
        (revenue - LAG(revenue, 1, NULL) OVER (
            PARTITION BY category ORDER BY sale_month
        )) / LAG(revenue, 1, NULL) OVER (
            PARTITION BY category ORDER BY sale_month
        ) * 100,
    2) AS mom_growth_pct
FROM monthly_sales
ORDER BY category, sale_month;
```

---

### 5.2 LEAD() —— 取下一行的值

```sql
-- 语法：LEAD(列名, 偏移行数, 默认值)

-- 业务需求：查看每个用户两次购买之间的间隔天数（用户购买频率分析）
SELECT
    user_id,
    order_id,
    order_date,
    -- LEAD：取下一条订单的日期
    LEAD(order_date, 1) OVER (
        PARTITION BY user_id
        ORDER BY order_date ASC
    ) AS next_order_date,
    -- 用 DATEDIFF 计算与下次购买的间隔天数
    DATEDIFF(
        LEAD(order_date, 1) OVER (
            PARTITION BY user_id ORDER BY order_date
        ),
        order_date
    ) AS days_to_next_order
FROM orders
ORDER BY user_id, order_date;
```



```sql
LEAD(order_date)          -- 等价于
LEAD(order_date, 1)       -- 取下一行（往后第1行）
```

---

## 6. 聚合窗口函数

普通聚合函数（SUM、AVG 等）加上 `OVER()` 子句，就变成了**聚合窗口函数**，既能保留原始行，又能计算累计/移动聚合值

### 6.1 SUM() —— 累计求和（Running Total）

```sql
-- 业务需求：查看每个用户的订单明细，同时显示该用户的累计消费额
SELECT
    user_id,
    order_id,
    order_date,
    amount,
    -- 累计求和：按日期排序，计算到当前行为止的累计金额
    SUM(amount) OVER (
        PARTITION BY user_id      -- 每个用户独立累计
        ORDER BY order_date ASC   -- 按下单时间从早到晚累加
        -- 不写 ROWS/RANGE，默认为 RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_amount,
    -- 对比：不加 ORDER BY = 该用户的总金额（每行都一样）
    SUM(amount) OVER (
        PARTITION BY user_id      -- 只分区，不排序
    ) AS total_amount_of_user
FROM orders
ORDER BY user_id, order_date;
```

![image-20260315105120971](day06_%E7%AA%97%E5%8F%A3%E5%87%BD%E6%95%B0.assets/image-20260315105120971.png)

---

### 6.2 AVG() —— 移动平均（Moving Average）

```sql
-- 业务需求：计算每个类目每月的3个月移动平均营收
SELECT
    category,
    sale_month,
    revenue,
    -- 3个月移动平均：当前行 + 前2行，共3行的平均
    ROUND(
        AVG(revenue) OVER (
            PARTITION BY category
            ORDER BY sale_month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW  -- 明确指定窗口帧：前2行到当前行
        ),
    2) AS moving_avg_3m
FROM monthly_sales
ORDER BY category, sale_month;
```

---

### 6.3 COUNT() —— 累计计数

```sql
-- 业务需求：按时间统计平台累计下单用户数（新用户增长曲线）
SELECT
    order_date,
    user_id,
    -- 统计截止到当前日期，出现过的不同用户数（注意：COUNT不支持DISTINCT配合窗口函数）
    COUNT(order_id) OVER (
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- 从第一行累计到当前行
    ) AS cumulative_orders
FROM orders
ORDER BY order_date;
```

---

### 6.4 MAX() / MIN() —— 窗口内极值

```sql
-- 业务需求：查看每笔订单的金额 vs 同类目的最高/最低订单金额
SELECT
    category,
    order_id,
    amount,
    -- 同类目内的最大订单金额
    MAX(amount) OVER (PARTITION BY category) AS max_in_category,
    -- 同类目内的最小订单金额
    MIN(amount) OVER (PARTITION BY category) AS min_in_category,
    -- 当前订单与类目最高价的差距
    MAX(amount) OVER (PARTITION BY category) - amount AS gap_to_max
FROM orders
ORDER BY category, amount DESC;
```

---

## 7. 分布函数

### 7.1 PERCENT_RANK() —— 百分比排名

**公式：** `(rank - 1) / (rows - 1)`，返回值范围 [0, 1]

```sql
-- 业务需求：计算每笔订单在全站的消费金额百分位（越高越贵）
SELECT
    order_id,
    user_id,
    amount,
    -- PERCENT_RANK：该订单金额在所有订单中处于哪个百分位
    ROUND(
        PERCENT_RANK() OVER (ORDER BY amount ASC) * 100,
    2) AS pct_rank
FROM orders
ORDER BY amount;
-- pct_rank=0 表示最低，pct_rank=100 表示最高
```

---

### 7.2 CUME_DIST() —— 累积分布

**公式：** 小于等于当前值的行数 / 总行数，返回值范围 (0, 1]

```sql
-- 业务需求：分析消费金额的累积分布，判断80%的订单在哪个金额以下
SELECT
    order_id,
    amount,
    ROUND(CUME_DIST() OVER (ORDER BY amount ASC), 4) AS cume_dist_val
FROM orders
ORDER BY amount;
-- cume_dist_val=0.8 表示有80%的订单金额 <= 当前金额
```

---

## 8. 窗口帧（Frame）详解

窗口帧是**最容易被忽视但非常关键**的概念，它决定了"当前行看哪些邻居"。

### 8.1 基本语法

```sql
ROWS  BETWEEN <起点> AND <终点>
RANGE BETWEEN <起点> AND <终点>
```

**起点 / 终点的关键字：**

| 关键字 | 含义 |
|--------|------|
| `UNBOUNDED PRECEDING` | 分区的第一行 |
| `n PRECEDING` | 当前行往前 n 行 |
| `CURRENT ROW` | 当前行 |
| `n FOLLOWING` | 当前行往后 n 行 |
| `UNBOUNDED FOLLOWING` | 分区的最后一行 |

### 8.2 ROWS vs RANGE 的区别

```sql
-- ROWS：按物理行数计算（精确）
-- RANGE：按值的范围计算（有相同值时会把同值行都纳入）

-- 示例：两者差异对比
SELECT
    sale_month,
    revenue,
    -- ROWS：严格按前2行
    SUM(revenue) OVER (
        ORDER BY sale_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS sum_rows,
    -- RANGE：按值范围（对于日期类数据，RANGE更常见）
    SUM(revenue) OVER (
        ORDER BY sale_month
        RANGE BETWEEN INTERVAL '2' MONTH PRECEDING AND CURRENT ROW
    ) AS sum_range
FROM monthly_sales
WHERE category = '电子';
```

### 8.3 常用窗口帧模板

```sql
-- 模板1：从头累计到当前行（累计求和标准写法）
SUM(amount) OVER (
    PARTITION BY user_id
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)

-- 模板2：最近3行的移动平均（含当前行）
AVG(revenue) OVER (
    ORDER BY sale_month
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
)

-- 模板3：整个分区（等价于不加窗口帧的聚合）
SUM(amount) OVER (
    PARTITION BY user_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)

-- 模板4：当前行前后各1行（共3行的移动平均）
AVG(revenue) OVER (
    ORDER BY sale_month
    ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
)
```

---

## 9. NAMED WINDOW 命名窗口

当同一个窗口定义被多次使用时，可以用 `WINDOW` 子句命名，避免重复代码。

```sql
-- 不使用命名窗口（重复写 OVER 子句，冗余）
SELECT
    user_id, order_id, amount,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC),
    RANK()       OVER (PARTITION BY user_id ORDER BY amount DESC),
    DENSE_RANK() OVER (PARTITION BY user_id ORDER BY amount DESC)
FROM orders;

-- ✅ 使用命名窗口（更简洁，推荐）
SELECT
    user_id, order_id, amount,
    ROW_NUMBER() OVER w,
    RANK()       OVER w,
    DENSE_RANK() OVER w
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY amount DESC);  -- 在末尾定义窗口别名
```

---

## 10. 综合实战案例

### 案例一：用户 RFM 分析（消费分层）

> RFM = Recency（最近消费）+ Frequency（消费频次）+ Monetary（消费金额）

```sql
-- 步骤1：计算每个用户的 RFM 基础指标
WITH user_rfm AS (
    SELECT
        user_id,
        -- R：最近一次消费距今天数（越小越好）
        DATEDIFF('2024-04-01', MAX(order_date)) AS recency,
        -- F：总消费次数
        COUNT(order_id) AS frequency,
        -- M：总消费金额
        SUM(amount) AS monetary
    FROM orders
    GROUP BY user_id
),
-- 步骤2：对每个指标打分（用 NTILE 分成5桶，5分最好）
user_scores AS (
    SELECT
        user_id, recency, frequency, monetary,
        -- R分：recency 越小越好，所以倒序分桶
        NTILE(5) OVER (ORDER BY recency ASC)     AS r_score,
        -- F分：frequency 越大越好
        NTILE(5) OVER (ORDER BY frequency DESC)  AS f_score,
        -- M分：monetary 越大越好
        NTILE(5) OVER (ORDER BY monetary DESC)   AS m_score
    FROM user_rfm
)
-- 步骤3：合并分数，判断用户分层
SELECT
    user_id, recency, frequency, monetary,
    r_score, f_score, m_score,
    (r_score + f_score + m_score) AS total_score,
    CASE
        WHEN (r_score + f_score + m_score) >= 12 THEN '💎 重要价值客户'
        WHEN (r_score + f_score + m_score) >= 9  THEN '⭐ 重要保持客户'
        WHEN (r_score + f_score + m_score) >= 6  THEN '🔄 一般价值客户'
        ELSE '😴 低活跃客户'
    END AS user_level
FROM user_scores
ORDER BY total_score DESC;
```

---

### 案例二：同比/环比分析（月度营收趋势）

```sql
SELECT
    category,
    sale_month,
    revenue,
    -- 环比：与上个月对比
    LAG(revenue, 1) OVER (PARTITION BY category ORDER BY sale_month) AS last_month,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (PARTITION BY category ORDER BY sale_month))
        / LAG(revenue, 1) OVER (PARTITION BY category ORDER BY sale_month) * 100,
    2) AS mom_pct,  -- Month Over Month 环比增长率%
    -- 累计营收（年初至今，YTD）
    SUM(revenue) OVER (
        PARTITION BY category,
            SUBSTR(sale_month, 1, 4)  -- 按类目+年份分区，每年重新累计
        ORDER BY sale_month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS ytd_revenue
FROM monthly_sales
ORDER BY category, sale_month;
```

---

### 案例三：TopN 问题 —— 每个类目销售额 Top2 的订单

```sql
-- 方法：先用 ROW_NUMBER/DENSE_RANK 打排名，再用子查询过滤
SELECT
    category, order_id, user_id, amount, rnk
FROM (
    SELECT
        category,
        order_id,
        user_id,
        amount,
        -- DENSE_RANK：若想允许并列，用 DENSE_RANK（可能取出超过2条）
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY amount DESC
        ) AS rnk
    FROM orders
) ranked
WHERE rnk <= 2   -- 取每个类目前2名
ORDER BY category, rnk;
```

---

### 案例四：新老用户识别

```sql
-- 业务需求：判断每笔订单是该用户的"首单"还是"复购单"
SELECT
    user_id,
    order_id,
    order_date,
    amount,
    -- ROW_NUMBER 按每个用户的下单时间排序，序号=1的就是首单
    ROW_NUMBER() OVER (
        PARTITION BY user_id
        ORDER BY order_date ASC
    ) AS order_seq,
    CASE
        WHEN ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date ASC) = 1
        THEN '首单（新用户）'
        ELSE '复购单（老用户）'
    END AS order_type
FROM orders
ORDER BY user_id, order_date;
```

---

### 案例五：连续N天下单

```sql
-- 业务需求：找出连续下单超过2天的用户（判断用户活跃度）
-- 思路：若是连续日期，则 order_date - ROW_NUMBER() 的差值恒定

WITH daily_orders AS (
    -- 先去重：每个用户每天只保留一条记录
    SELECT DISTINCT user_id, order_date
    FROM orders
),
flagged AS (
    SELECT
        user_id,
        order_date,
        -- 用日期减去排名，若连续则差值相同（形成分组标志）
        DATE_SUB(order_date, INTERVAL
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date)
        DAY) AS grp_flag
    FROM daily_orders
)
SELECT
    user_id,
    MIN(order_date) AS streak_start,  -- 连续开始日期
    MAX(order_date) AS streak_end,    -- 连续结束日期
    COUNT(*) AS consecutive_days      -- 连续天数
FROM flagged
GROUP BY user_id, grp_flag
HAVING consecutive_days >= 2          -- 筛选连续2天及以上
ORDER BY user_id, streak_start;
```

---

## 11. 注意事项与常见错误

### 11.1 执行顺序

```
SQL 执行顺序：
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → WINDOW → ORDER BY → LIMIT

⚠️ 关键点：窗口函数在 SELECT 阶段执行，因此：
  ✅ WHERE 中不能直接使用窗口函数（因为 WHERE 先于 SELECT 执行）
  ✅ 必须用子查询或 CTE 包裹后再过滤
```

```sql
-- ❌ 错误写法：不能在 WHERE 中直接使用窗口函数
SELECT user_id, amount, ROW_NUMBER() OVER (...) AS rn
FROM orders
WHERE rn = 1;  -- 报错！WHERE 执行时 rn 还不存在

-- ✅ 正确写法：用子查询包裹
SELECT * FROM (
    SELECT user_id, amount,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
    FROM orders
) t
WHERE t.rn = 1;

-- ✅ 也可以用 CTE（更清晰）
WITH ranked AS (
    SELECT user_id, amount,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
    FROM orders
)
SELECT * FROM ranked WHERE rn = 1;
```

### 11.2 其他注意事项

```sql
-- ⚠️ 注意1：窗口函数不支持 DISTINCT
-- ❌ 以下写法报错
COUNT(DISTINCT user_id) OVER (PARTITION BY category)

-- ⚠️ 注意2：ORDER BY 影响聚合窗口函数的默认窗口帧
-- 加了 ORDER BY 后，SUM() 的默认帧变为 RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW（累计）
-- 不加 ORDER BY，默认帧为 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING（全分区）

-- ⚠️ 注意3：NULL 值的处理
-- LAG/LEAD 在边界行（第一行/最后一行）返回 NULL，需指定默认值
LAG(revenue, 1, 0) OVER (...)  -- 第三个参数为默认值，没有上一行时返回0

-- ⚠️ 注意4：MySQL 版本要求
-- 窗口函数需要 MySQL 8.0 及以上版本，5.x 不支持！
-- 可通过 SELECT VERSION(); 查看版本
```

---

## 12. 面试高频题

| 题目类型 | 核心函数 | 关键思路 |
|---------|---------|---------|
| 每组取 Top-N | `ROW_NUMBER` / `DENSE_RANK` | 先打排名，子查询过滤 `rn <= N` |
| 计算环比/同比 | `LAG` / `LEAD` | 用偏移函数取上期值，再计算差值 |
| 累计求和/累计用户数 | `SUM() OVER` | 加 `ORDER BY`，默认帧即为累计 |
| 移动平均 | `AVG() OVER` + `ROWS BETWEEN` | 指定 `ROWS BETWEEN n PRECEDING AND CURRENT ROW` |
| 中位数 | `PERCENT_RANK` / `NTILE` | 找 `pct_rank` 接近 0.5 的值 |
| 连续登录/下单 N 天 | `ROW_NUMBER` + 日期差 | 经典"日期 - 序号 = 常数"分组法 |
| 用户首单识别 | `ROW_NUMBER` | 按用户+时间排序，`rn=1` 即首单 |
| 消费分层（RFM） | `NTILE` | 对 R/F/M 分别打分再汇总 |

---

> 📌 **学习建议：**
> 1. 先把建表语句跑通，再逐节练习
> 2. 重点掌握：`ROW_NUMBER`、`RANK`/`DENSE_RANK`、`LAG`/`LEAD`、`SUM() OVER`
> 3. 理解执行顺序（窗口函数在 WHERE 之后），避免低级错误
> 4. 多练综合案例，面试中窗口函数几乎必考
