# 01 insert语句

本文以下面这张 `students` 表作为贯穿全文的演示表：

```sql
CREATE TABLE students (
    id         INT           NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(50)   NOT NULL,
    age        TINYINT       UNSIGNED,
    gender     ENUM('男','女','保密') DEFAULT '保密',
    email      VARCHAR(150)  DEFAULT NULL,
    score      DECIMAL(5,2)  NOT NULL DEFAULT 0.00,
    is_active  TINYINT(1)    NOT NULL DEFAULT 1,
    created_at DATETIME      DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```



> * `INSERT` 语句用于向表中写入新的数据行。它是数据进入数据库的唯一入口，语法有多种形式，适用于不同的场景
> * 括号里的列名和值必须**一一对应**，数量相同，顺序一致。列名的顺序不必与建表时的顺序相同，MySQL 会按照写的列名顺序来匹配值

```sql
INSERT INTO 表名 (列1, 列2, 列3, ...) VALUES (值1, 值2, 值3, ...);
```



> * 只要被省略的列有默认值或允许为 `NULL`，就可以不写，MySQL 会自动使用默认值填充

```sql
-- 只插入必填的 name，其余全部使用默认值
INSERT INTO students (name) VALUES ('李四');

-- 插入 name 和 score，其余使用默认值
INSERT INTO students (name, score) VALUES ('王五', 95.00);
```



> * `VALUES` 后面可以跟多组括号，用逗号分隔，一次性插入多行。这比多条单独的 `INSERT` 语句**效率高得多**，因为只需要一次网络往返和一次事务提交
> * 批量插入时，任意一行数据有问题（如违反约束），整条语句都会失败回滚，所有行都不会插入

```sql
INSERT INTO students (name, age, gender, score)
VALUES
    ('孙七',   19, '男', 91.00),
    ('周八',   21, '女', 83.50),
    ('吴九',   20, '男', 67.00),
    ('郑十',   23, '女', 55.00),
    ('陈十一', 18, '男', 100.00);
```

# 02 update语句

`UPDATE` 语句用于修改表中已有行的数据。它是生产环境中最需要谨慎操作的语句之一，因为一旦执行，数据立即被覆盖，如果忘记写 `WHERE` 条件，将会修改**整张表所有行的数据**



## 02.1 基本语法

> * `SET` 子句可以同时修改多列，用逗号分隔。`WHERE` 子句指定要修改哪些行，**强烈建议每次 UPDATE 都写 WHERE**，否则就是全表更新

```sql
UPDATE 表名
SET 列1 = 新值1, 列2 = 新值2, ...
WHERE 条件;
```



## 02.2 更新单列

```sql
-- 把 id = 1 的学生成绩改为 95.00
UPDATE students SET score = 95.00 WHERE id = 1;

-- 把姓名为"张三"的学生邮箱更新
UPDATE students SET email = 'zhangsan_new@example.com' WHERE name = '张三';
```



## 02.3 更新多列

```sql
-- 把 id = 2 的学生年龄改为 21，成绩改为 88.00，邮箱也更新
UPDATE students
SET age = 21, score = 88.00, email = 'lisi@example.com'
WHERE id = 2;
```

# 03 delete语句

`DELETE` 语句用于删除表中的数据行。与 `UPDATE` 一样，它是高危操作，忘记写 `WHERE` 条件会导致**全表数据被清空**，且逐行删除的过程无法中断



> * 基本语法

```sql
DELETE FROM 表名 WHERE 条件;
```



> * 删除单行

```sql
-- 删除 id = 1 的学生记录
DELETE FROM students WHERE id = 1;
```



> * 删除多行

```sql
-- 删除所有成绩低于 60 分的学生
DELETE FROM students WHERE score < 60;

-- 删除所有已停用（is_active = 0）的学生
DELETE FROM students WHERE is_active = 0;

-- 删除年龄小于 18 岁且成绩低于 50 分的学生
DELETE FROM students WHERE age < 18 AND score < 50;
```



> * 使用`in`删除多个指定行

```sql
-- 删除 id 为 3、5、7 的三条记录
DELETE FROM students WHERE id IN (3, 5, 7);

-- 删除姓名在指定列表中的记录
DELETE FROM students WHERE name IN ('张三', '李四', '王五');
```

# 04 select语句（单表）

`SELECT` 是 SQL 中功能最强大、使用频率最高的语句。从最简单的全表查询到复杂的多表关联、分组聚合，都依赖 `SELECT` 来完成



## 04.1 基本语法

> * 这些子句的**书写顺序是固定的**，不能随意调换，但每个子句都是可选的（除了 `SELECT` 和 `FROM`）。它们的**执行顺序**与书写顺序不同，MySQL 内部执行顺序为：
>
>    `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`
>
>    理解这个执行顺序非常重要，它解释了为什么 `WHERE` 里不能用 `SELECT` 中定义的别名，而 `ORDER BY` 里可以
>
>    ==WHERE 是在分组之前过滤原始行；HAVING 是在分组之后过滤分组结果==
>
>    select检索多个列，列之间必须用逗号分隔，最后一个列名后不要加
>    
>  *  ==HAVING 能用 SELECT 的别名，不是因为 SELECT 提前执行了，而是 MySQL 的解析器非常聪明，帮你做了一个“语法糖”式的底层替换，让你少写几次重复的计算公式==

```sql
SELECT 列名1, 列名2, ...
FROM 表名
WHERE 条件
GROUP BY 分组列
HAVING 分组后条件
ORDER BY 排序列 ASC|DESC
LIMIT 偏移量, 行数;
```

## 04.2 查询所有列和指定列

> * `SELECT *` 虽然方便，但在实际开发中应尽量避免，原因是它会把所有列都传输到应用层，浪费网络带宽和内存，尤其是表中有 `TEXT`、`BLOB` 等大字段时性能影响明显

```sql
-- 查询所有列（生产环境不推荐，性能差，容易查出不需要的敏感列）
SELECT * FROM students;

-- 查询指定列（推荐，只取需要的数据）
SELECT id, name, score FROM students;
```

## 04.3 WHERE子句

`WHERE` 子句用来筛选满足条件的行，是使用频率最高的子句之一

==聚合函数（如 COUNT、SUM、AVG 等）绝对不能用在 WHERE 里==

### 1 比较运算符

> * 当对字符串进行运算比较时，需要使用单引号，数值列则不需要

```sql
-- 等于
SELECT * FROM students WHERE gender = '男';

-- 不等于（两种写法等价）
SELECT * FROM students WHERE gender != '女';
SELECT * FROM students WHERE gender <> '女';

-- 大于、小于、大于等于、小于等于
SELECT * FROM students WHERE score >= 90;
SELECT * FROM students WHERE age < 20;
SELECT * FROM students WHERE score BETWEEN 60 AND 90;  -- 包含两端，等价于 score >= 60 AND score <= 90
```

###2 逻辑运算符

```sql
-- AND：两个条件同时满足
SELECT * FROM students WHERE gender = '男' AND score >= 80;

-- OR：满足任意一个条件
SELECT * FROM students WHERE score < 60 OR is_active = 0;

-- NOT：对条件取反
SELECT * FROM students WHERE NOT gender = '男';

-- 混合使用时，AND 优先级高于 OR，建议用括号明确优先级
SELECT * FROM students WHERE (gender = '男' OR gender = '女') AND score >= 70;
```

### 3 IN —— 匹配多个值

> * `in`和`or`其实功能上是一样的
> * 也可以多值匹配，比如说（xxx,yyy） in (        )，就是匹配一个组合

```sql
-- IN：列的值属于给定集合中的任意一个
SELECT * FROM students WHERE age IN (18, 19, 20);

-- NOT IN：列的值不在给定集合中
SELECT * FROM students WHERE gender NOT IN ('男', '女');
```

### 4 LIKE —— 模糊匹配

可以与not 字段配合使用，表示否定查询

```sql
-- % 匹配任意数量的字符（包括零个）
SELECT * FROM students WHERE name LIKE '张%';    -- 姓张的学生
SELECT * FROM students WHERE email LIKE '%@gmail.com';  -- 使用 gmail 的学生
SELECT * FROM students WHERE name LIKE '%三%';   -- 名字里含有"三"的学生

-- _ 匹配任意单个字符
SELECT * FROM students WHERE name LIKE '张_';    -- 姓张且名字共两个字的学生
```

一次性满足多个字符的匹配可以这样写：

```sql
-- 要求name里同时有aa和bb两个
where name like '%aa%bb%'  -- 这里有个注意点，只会查询先aa后bb的
-- 或者这样写
where name like '%aa%' and name like '%bb%'
```

### 5 IS NULL

```sql
-- 查询 email 为空的学生（NULL 不能用 = 比较，必须用 IS NULL）
SELECT * FROM students WHERE email IS NULL;

-- 查询 email 不为空的学生
SELECT * FROM students WHERE email IS NOT NULL;
```

## 04.4 ORDER BY子句

> * ==不一定非要select后的列才能用于order by ，用非检索的列来排序也是可以的==
> * 多列排序是按顺序来，先按第一个排完，然后按第二个的排，……
> * 排序关键字（desc、asc）只会作用于它的前一个列名，所以如果是有多种不同顺序，要在每个列名后写明怎么排

```sql
-- 按 score 升序排列（ASC 是默认值，可以省略）
SELECT id, name, score FROM students ORDER BY score ASC;

-- 按 score 降序排列
SELECT id, name, score FROM students ORDER BY score DESC;

-- 多列排序：先按 score 降序，score 相同时再按 name 升序
SELECT id, name, score FROM students ORDER BY score DESC, name ASC;
```

## 04.5 group by子句

> * `GROUP BY` 是 SQL 中用于**分组统计**的核心子句。它把表中的行按照指定列的值归类，将相同值的行划分到同一个组里，然后对每个组分别进行聚合计算
> * 当分组列中含有 `NULL` 值时，MySQL 会把所有 `NULL` 值的行**归为同一组**，单独计算，而不是忽略它们。

```sql
-- 有了 GROUP BY，一条语句搞定所有分组
SELECT gender, AVG(score) AS 平均分
FROM students
GROUP BY gender;
-- GROUP BY 存在的核心价值：让数据库自动发现所有分组，并对每个分组独立计算
```

### 1 按单列分组

```sql
-- 按性别分组，统计每个性别的人数
SELECT gender, COUNT(*) AS 人数
FROM students
GROUP BY gender;

-- 按年龄分组，统计每个年龄段有多少学生，以及该年龄段的平均分
SELECT
    age,
    COUNT(*)          AS 人数,
    ROUND(AVG(score), 2) AS 平均分
FROM students
GROUP BY age
ORDER BY age ASC;

-- 按 is_active 分组，统计激活和未激活的学生各有多少
SELECT
    is_active,
    COUNT(*) AS 人数
FROM students
GROUP BY is_active;
```

### 2 按多列分组

`GROUP BY` 后面可以跟**多个列名**，用逗号分隔。此时 MySQL 会把这几列的值的**组合**视为分组依据，只有所有列的值都相同的行，才会被划入同一个组

```sql
-- 按性别和是否激活两列组合分组
SELECT
    gender,
    is_active,
    COUNT(*)             AS 人数,
    ROUND(AVG(score), 2) AS 平均分
FROM students
GROUP BY gender, is_active
ORDER BY gender, is_active;
```

==GROUP BY 后，未聚合字段（quantity, price）没有确定规则，MySQL只能随机选一个值，不能保证和前面的对应==

## 04.6 having -- 分组后过滤

`HAVING` 是专门为 `GROUP BY` 设计的过滤子句，用于在分组之后对每个组的聚合结果进行筛选

```sql
-- 找出人数超过 1 人的性别分组
SELECT gender, COUNT(*) AS 人数
FROM students
GROUP BY gender
HAVING COUNT(*) > 1;

-- 找出平均成绩高于 75 分的性别分组
SELECT gender, ROUND(AVG(score), 2) AS 平均分
FROM students
GROUP BY gender
HAVING AVG(score) > 75
ORDER BY 平均分 DESC;

-- 找出总消费金额超过 100 元的学生
SELECT
    student_id,
    ROUND(SUM(quantity * unit_price - discount), 2) AS 消费总额  -- ROUND()函数控制小数位数
FROM orders
GROUP BY student_id
HAVING SUM(quantity * unit_price - discount) > 100
ORDER BY 消费总额 DESC;
```
## 04.7 distinct关键字

> * `DISTINCT` 用来**去重**：把查询结果中**重复的行**合并成一行返回
> * 必须放在列名的前面

```sql
-- 只取某一列的唯一值
SELECT DISTINCT city
FROM users;
-- 含义：返回 city 列中所有不同的城市（重复城市只出现一次）。

-- 多列去重：按“整行组合”去重
SELECT DISTINCT city, gender
FROM users;
-- 这不是分别对 city、gender 各自去重，而是对 (city, gender) 这个组合去重。
-- 例如：(Beijing, M) 和 (Beijing, F) 会保留两行，因为组合不同。
```

# 05 limit 分页查询

在实际开发中，当一张表的数据量达到成千上万、甚至千万级别时，如果用 `SELECT *` 一次性把所有数据查出来，不仅会导致数据库服务器内存溢出、网络带宽耗尽，前端页面也会因为渲染数据量过大而直接崩溃。

**分页查询（Pagination）**就是为了解决这个问题而生的。它允许我们每次只向数据库请求“一页”的数据，按需加载。这不仅是提升用户体验（UX）的核心手段，更是保护数据库性能的底层防线

```sql
-- 写法一：明确指定 LIMIT 和 OFFSET（推荐，语义最清晰，属于 SQL 标准）
SELECT 列名 FROM 表名 LIMIT 返回行数 OFFSET 开始行;

-- 写法二：简写模式（MySQL 特有，使用极其广泛）
SELECT 列名 FROM 表名 LIMIT 开始行, 返回行数;
```

**开始行（Offset）**：你希望跳过前面多少条数据，从哪一条开始取。**注意：开始行是从 0 开始计算的**

**返回行数（Limit/PageSize）**：你希望这一页最多显示几条数据



```sql
-- 获取第 1 页数据（跳过 0 条，取 5 条）
SELECT id, product, unit_price, created_at 
FROM orders 
ORDER BY created_at DESC 
LIMIT 5 OFFSET 0;   -- 或者简写为 LIMIT 0, 5 或直接 LIMIT 5

-- 获取第 2 页数据（跳过前 5 条，取 5 条）
SELECT id, product, unit_price, created_at 
FROM orders 
ORDER BY created_at DESC 
LIMIT 5 OFFSET 5;   -- 或者简写为 LIMIT 5, 5

-- 获取第 3 页数据（跳过前 10 条，取 5 条）
SELECT id, product, unit_price, created_at 
FROM orders 
ORDER BY created_at DESC 
LIMIT 5 OFFSET 10;  -- 或者简写为 LIMIT 10, 5
```

















