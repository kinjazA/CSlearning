# 01 统计函数

## 01.1 count函数

`COUNT` 是 SQL 中最常用的聚合函数之一，专门用于**统计行数或非空值的数量**



以下面这张 `students` 表作为演示：

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

INSERT INTO students (name, age, gender, email, score, is_active) VALUES
    ('张三',   20, '男',   'zhangsan@example.com', 88.00, 1),
    ('李四',   21, '女',   'lisi@example.com',     76.00, 1),
    ('王五',   19, '男',   NULL,                   91.00, 1),
    ('赵六',   22, '女',   NULL,                   55.00, 0),
    ('孙七',   NULL,'男',  'sunqi@example.com',    63.00, 1),
    ('周八',   20, '保密', NULL,                   47.00, 0),
    ('吴九',   21, '女',   'wujiu@example.com',    95.00, 1),
    ('郑十',   19, '男',   NULL,                   82.00, 1);
```

### 1 COUNT(*) 

> * `COUNT(*)` 统计表中**所有行的数量**，不管某一行的列值是什么，哪怕整行全是 `NULL`，也会被计入

```sql
SELECT COUNT(*) FROM students;
-- 结果为8
```

### 2 COUNT(列名) 

```sql
SELECT COUNT(email) FROM students;
-- 结果为4
-- COUNT(列名) 统计该列中值不为 NULL 的行数。在测试数据中，email 有 4 行不为 NULL，所以结果是 4，而不是 8

SELECT COUNT(age) FROM students;
-- 结果：7
-- 因为孙七的 age 是 NULL，所以只统计到 7 行
```

可以用来**快速统计某列有多少行填写了数据、有多少行是空的**：

```sql
SELECT
    COUNT(*)      AS 总行数,
    COUNT(email)  AS 填写了邮箱的人数,
    COUNT(*) - COUNT(email) AS 未填写邮箱的人数,
    COUNT(age)    AS 填写了年龄的人数
FROM students;
```

![image-20260303083751436](F:\note\MySQL\day04_函数.assets\image-20260303083751436.png)



### 3 COUNT(DISTINCT 列名)

`COUNT(DISTINCT 列名)` 会先对该列去重，去掉 `NULL`，然后再统计数量。它回答的问题是：**这一列一共出现了多少种不同的值？**

```sql
SELECT COUNT(DISTINCT gender) FROM students;
-- 结果：3（男、女、保密三种值）

SELECT COUNT(DISTINCT age) FROM students;
-- 结果：3（19、20、21 三个不重复的年龄，NULL 和重复值都被排除）

-- 还可以对多列组合去重：
-- 统计不重复的 (gender, age) 组合有多少种
SELECT COUNT(DISTINCT gender, age) FROM students;
```

![image-20260303084017117](F:\note\MySQL\day04_函数.assets\image-20260303084017117.png)

## 01.2 sum函数

`SUM` 是 SQL 中专门用于**求和**的聚合函数，它对指定列中所有非 NULL 的数值进行累加，返回总和

```sql
-- 新建订单表
CREATE TABLE orders (
    id          INT            NOT NULL AUTO_INCREMENT PRIMARY KEY,
    student_id  INT            NOT NULL,
    product     VARCHAR(100)   NOT NULL,
    quantity    INT            NOT NULL DEFAULT 1,
    unit_price  DECIMAL(10,2)  NOT NULL,
    discount    DECIMAL(4,2)   NOT NULL DEFAULT 0.00,
    is_paid     TINYINT(1)     NOT NULL DEFAULT 0,
    created_at  DATETIME       DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 插入测试数据
INSERT INTO orders (student_id, product, quantity, unit_price, discount, is_paid) VALUES
    (1, 'Python教材',   2, 59.00,  0.00, 1),
    (1, '笔记本',       5, 12.00,  0.00, 1),
    (2, 'Java教材',     1, 79.00, 10.00, 1),
    (2, '钢笔',         3,  8.00,  0.00, 0),
    (3, 'MySQL教材',    1, 69.00,  5.00, 1),
    (3, '荧光笔',       4,  6.50,  0.00, 1),
    (4, '算法教材',     2, 89.00, 15.00, 0),
    (5, 'Linux教材',    1, 75.00,  0.00, 1),
    (5, '笔记本',       3, 12.00,  0.00, 1),
    (6, 'Python教材',   1, 59.00,  5.00, 0),
    (7, 'Java教材',     2, 79.00,  0.00, 1),
    (8, '数据结构教材', 1, 85.00, 10.00, 1);
```



`SUM `接受一个数值列或数值表达式作为参数，返回该列所有非 NULL 值的总和。如果表中没有任何行，或者该列所有值都是 `NULL`，则 `SUM` 返回 `NULL` 而不是 0

```sql
-- 基本语法
SELECT SUM(列名) FROM 表名;
```

### 1 对单列求和

```sql
-- 统计所有订单的总金额（unit_price 之和）
SELECT SUM(unit_price) AS 总单价之和 FROM orders;
-- 结果：756.50（所有行的 unit_price 相加）

-- 统计所有订单的总数量
SELECT SUM(quantity) AS 总数量 FROM orders;
-- 结果：27
```

### 2  对表达式求和

`SUM` 的参数不仅可以是列名，还可以是一个**表达式**。这是 `SUM` 最体现价值的地方，因为实际业务中往往需要先计算再求和，比如"总金额 = 数量 × 单价 - 折扣"

```sql
-- 计算每条订单的实际金额 = quantity * unit_price - discount
-- 然后用 SUM 求所有订单的实际总金额
SELECT SUM(quantity * unit_price - discount) AS 实际总金额
FROM orders;
-- 这条语句的执行逻辑是：对每一行先计算 quantity * unit_price - discount 的值，得到该行的实际金额，然后把所有行的实际金额加在一起

-- 计算折扣总额（所有订单一共优惠了多少钱）
SELECT SUM(discount) AS 总折扣金额 FROM orders;

-- 计算如果没有折扣，原本的总金额是多少
SELECT SUM(quantity * unit_price) AS 原始总金额 FROM orders;

-- 同时输出原始总金额、总折扣、实际总金额，形成一张汇总表
SELECT
    SUM(quantity * unit_price)            AS 原始总金额,
    SUM(discount)                          AS 总折扣金额,
    SUM(quantity * unit_price - discount)  AS 实际总金额
FROM orders;
```

### 3 与 WHERE 结合

```sql
-- 只统计已支付订单（is_paid = 1）的总金额
SELECT SUM(quantity * unit_price - discount) AS 已支付总金额
FROM orders
WHERE is_paid = 1;

-- 统计单价超过 60 元的商品的总销售金额
SELECT SUM(quantity * unit_price) AS 高价商品总金额
FROM orders
WHERE unit_price > 60;
```

### 4 与 GROUP BY 结合

```sql
-- 按学生分组，统计每个学生的总消费金额
SELECT
    student_id,
    SUM(quantity * unit_price - discount) AS 消费总额
FROM orders
GROUP BY student_id
ORDER BY 消费总额 DESC;

-- 按商品分组，统计每种商品的总销售数量和总销售金额
SELECT
    product                               AS 商品名称,
    SUM(quantity)                          AS 总销量,
    SUM(quantity * unit_price - discount)  AS 总销售金额
FROM orders
GROUP BY product
ORDER BY 总销售金额 DESC;
```

## 01.3 avg函数

`AVG` 是 SQL 中专门用于计算**平均值**的聚合函数，全称是 Average。==它将指定列中所有非 NULL 的数值加在一起，除以非 NULL 值的数量==，得到算术平均值

### 1 对单列求平均

```sql
-- 计算所有学生的平均成绩
SELECT AVG(score) AS 平均成绩 FROM students;
-- 结果：74.625（8 个学生的成绩总和除以 8）

-- 计算所有学生的平均年龄
SELECT AVG(age) AS 平均年龄 FROM students;
-- 注意：孙七的 age 是 NULL，所以这里是 7 个有年龄数据的学生的总年龄除以 7，而不是除以 8。这是 AVG 对 NULL 的处理规则
```

### 2 对表达式求平均

与 `SUM` 一样，`AVG` 的参数不仅可以是列名，还可以是一个表达式，MySQL 会先对每一行计算表达式的值，再对这些值求平均

```sql
-- 计算每笔订单实际金额的平均值（每笔平均消费多少）
SELECT AVG(quantity * unit_price - discount) AS 平均订单金额
FROM orders;

-- 计算平均折扣率（折扣占原价的百分比）
SELECT AVG(discount / (quantity * unit_price) * 100) AS 平均折扣率
FROM orders
WHERE discount > 0;   -- 只统计有折扣的订单
```

# 02 字符串函数

## 02.1 LENGTH函数

`LENGTH` 返回字符串占用的**字节数**，而不是字符数。对于纯英文和数字，一个字符占 1 个字节；对于中文字符，在 `utf8mb4` 编码下一个汉字占 **3 个字节**

```sql
SELECT LENGTH('hello');         -- 结果：5（5个英文字母，每个1字节）
SELECT LENGTH('你好');           -- 结果：6（2个汉字，每个3字节）
SELECT LENGTH('hello你好');      -- 结果：11（5个英文 + 2个汉字）
SELECT LENGTH('');              -- 结果：0（空字符串）
SELECT LENGTH(NULL);            -- 结果：NULL

-- 实际应用：找出姓名字节长度超过 9 的学生（即名字超过 3 个汉字）
SELECT name, LENGTH(name) AS 字节长度
FROM students
WHERE LENGTH(name) > 9;
```

## 02.2 CHAR_LENGTH函数

`CHAR_LENGTH` 返回字符串的**字符个数**，不管每个字符占几个字节，一个字符就计数为 1。处理中文等多字节字符时，`CHAR_LENGTH` 比 `LENGTH` 更符合直觉，是**实际开发中更常用的函数**

```sql
SELECT CHAR_LENGTH('hello');      -- 结果：5
SELECT CHAR_LENGTH('你好');        -- 结果：2（2个汉字，就是2个字符）
SELECT CHAR_LENGTH('hello你好');   -- 结果：7（5个英文 + 2个汉字）
SELECT CHAR_LENGTH('');           -- 结果：0
SELECT CHAR_LENGTH(NULL);         -- 结果：NULL

-- 找出姓名超过 3 个字符的学生
SELECT name, CHAR_LENGTH(name) AS 字符数
FROM students
WHERE CHAR_LENGTH(name) > 3;
```

## 02.3 UPPER / UCASE 

```sql
SELECT UPPER('hello world');    -- 结果：HELLO WORLD
SELECT UPPER('Hello123');       -- 结果：HELLO123（数字不受影响）
SELECT UPPER('你好hello');       -- 结果：你好HELLO（中文不受影响）

-- 实际应用：统一将邮箱转为大写后比较（不区分大小写的搜索）
SELECT name, email
FROM students
WHERE UPPER(email) = UPPER('ZhangSan@Example.COM');
```

## 02.4 LOWER / LCASE

```sql
SELECT LOWER('HELLO WORLD');    -- 结果：hello world
SELECT LOWER('Hello123');       -- 结果：hello123

-- 实际应用：存入数据库前统一将邮箱转为小写，避免大小写不一致
UPDATE students SET email = LOWER(email);
```

## 02.5 CONCAT函数

`CONCAT` 把多个字符串拼接成一个。可以接受任意数量的参数。**只要有任意一个参数是 NULL，结果就是 NULL**，这是最需要注意的地方

```sql
-- 基本语法
SELECT CONCAT(str1, str2, str3, ...);

SELECT CONCAT('hello', ' ', 'world');        -- 结果：hello world
SELECT CONCAT('姓名：', '张三');              -- 结果：姓名：张三
SELECT CONCAT('hello', NULL, 'world');       -- 结果：NULL（含 NULL 则结果为 NULL）

-- 实际应用：把姓名和成绩拼成一句话输出
SELECT CONCAT(name, ' 的成绩是 ', score, ' 分') AS 成绩描述
FROM students;
```

## 02.6 CONCAT_WS函数

`CONCAT_WS` 全称是 Concatenate With Separator，用**统一的分隔符**把多个字符串连接起来。与 `CONCAT` 的重要区别在于：**`CONCAT_WS` 会自动忽略 NULL 参数**，不会因为某个参数为 NULL 就返回 NULL

```sql
-- 基本语法
SELECT CONCAT_WS(分隔符, str1, str2, str3, ...);

SELECT CONCAT_WS('-', '2026', '03', '04');          -- 结果：2026-03-04
SELECT CONCAT_WS(', ', '张三', '李四', '王五');      -- 结果：张三, 李四, 王五
SELECT CONCAT_WS(', ', '张三', NULL, '王五');        -- 结果：张三, 王五（NULL 被忽略）
SELECT CONCAT_WS(NULL, '张三', '李四');              -- 结果：NULL（分隔符本身为 NULL 才返回 NULL）

-- 实际应用：把姓名、性别、成绩用"|"分隔拼在一起，生成导出格式
SELECT CONCAT_WS(' | ', name, gender, score) AS 导出信息
FROM students;
```

## 02.7 SUBSTRING / SUBSTR

`SUBSTRING` 从指定位置开始截取子字符串。==**位置从 1 开始计数**，不是从 0 开始==，这和很多编程语言不同，需要特别注意。`SUBSTR` 是 `SUBSTRING` 的简写，两者完全等价

```sql
-- 基本语法
SELECT SUBSTRING(str, 起始位置);
SELECT SUBSTRING(str, 起始位置, 截取长度);

-- 起始位置也可以是负数，表示从字符串末尾往前数：
SELECT SUBSTRING('hello world', -5);       -- 结果：world（从倒数第5个字符截到末尾）
SELECT SUBSTRING('hello world', -5, 3);    -- 结果：wor（从倒数第5位截取3个字符）

-- 实际应用：从邮箱中截取用户名部分（@ 符号之前的内容）
-- 这里先用 LOCATE 找到 @ 的位置，再用 SUBSTRING 截取
SELECT
    email,
    SUBSTRING(email, 1, LOCATE('@', email) - 1) AS 用户名
FROM students
WHERE email IS NOT NULL;
```

## 02.8 LOCATE / POSITION / INSTR

`LOCATE` 返回子串在字符串中**第一次出现的位置**（从 1 开始计数）。如果子串不存在，返回 `0`。`POSITION(子串 IN 字符串)` 和 `INSTR(字符串, 子串)` 与 `LOCATE` 功能相同，只是写法不同

```sql
-- 基本语法
SELECT LOCATE(子串, 字符串);
SELECT LOCATE(子串, 字符串, 起始位置);   -- 从指定位置开始查找

-- 结合 SUBSTRING 提取 @ 之前的用户名
SELECT
    email,
    SUBSTRING(email, 1, LOCATE('@', email) - 1) AS 用户名,
    SUBSTRING(email, LOCATE('@', email) + 1)    AS 域名
FROM students
WHERE email IS NOT NULL;
```

## 02.9 REPLACE 

`REPLACE` 把字符串中**所有**匹配的子串都替换成新的字符串。注意它是**大小写敏感**的

```sql
-- 基本语法
SELECT REPLACE(str, 被替换的子串, 替换成的子串);

-- 实际应用：批量修正数据，把邮箱中的旧域名替换为新域名
UPDATE students
SET email = REPLACE(email, '@old-domain.com', '@new-domain.com')
WHERE email LIKE '%@old-domain.com';

-- 实际应用：去除字符串中的空格（把所有空格替换为空字符串）
SELECT REPLACE('hello world test', ' ', '');   -- 结果：helloworldtest
```

## 02.10 LEFT/RIGHT

```sql
-- 基本语法

-- 从字符串左边（开头）截取指定数量的字符，相当于 SUBSTRING(str, 1, 长度)
SELECT LEFT(str, 长度);

-- 从字符串右边（末尾）截取指定数量的字符
SELECT RIGHT(str, 长度);

-- 实际应用：取手机号前 3 位（运营商识别码）
SELECT
    name,
    LEFT(phone, 3) AS 运营商前缀
FROM students
WHERE phone IS NOT NULL;

-- 实际应用：取邮箱后缀（@ 之后的域名部分）
SELECT
    email,
    RIGHT(email, CHAR_LENGTH(email) - LOCATE('@', email)) AS 邮箱域名
FROM students
WHERE email IS NOT NULL;
```

# 03 数学函数

## 03.1 ROUND —— 四舍五入

`ROUND` 是最常用的取整函数，按照数学上的四舍五入规则处理数值。第二个参数指定保留的小数位数，可以是正数、零或负数

```sql
-- 基本语法
SELECT ROUND(数值);             -- 四舍五入到整数
SELECT ROUND(数值, 小数位数);   -- 四舍五入到指定小数位数

-- 指定小数位数
SELECT ROUND(3.14159, 2);      -- 结果：3.14（保留2位小数）
SELECT ROUND(3.14159, 4);      -- 结果：3.1416（保留4位小数）
SELECT ROUND(3.145, 2);        -- 结果：3.15（第3位是5，进位）

-- 小数位数为负数：对整数部分进行四舍五入
SELECT ROUND(1567.89, -1);     -- 结果：1570（四舍五入到十位）
SELECT ROUND(1567.89, -2);     -- 结果：1600（四舍五入到百位）
SELECT ROUND(1567.89, -3);     -- 结果：2000（四舍五入到千位）

-- 负数的四舍五入
SELECT ROUND(-3.5);            -- 结果：-4
SELECT ROUND(-3.14159, 2);     -- 结果：-3.14

-- 实际应用：计算每个学生的平均订单金额，保留2位小数
SELECT
    student_id,
    ROUND(AVG(quantity * unit_price - discount), 2) AS 平均消费金额
FROM orders
GROUP BY student_id
ORDER BY 平均消费金额 DESC;
```

## 03.2 CEIL / CEILING —— 向上取整

`CEIL` 总是返回**大于或等于**该数值的最小整数，也就是无论小数部分是多少，只要有小数就向上进一位。在分页计算、资源分配等场景中非常常用

```sql
-- 基本语法
SELECT CEIL(数值);
SELECT CEILING(数值);   -- 与 CEIL 完全等价

-- 实际应用：分页计算总页数
-- 假设有 25 条记录，每页展示 10 条，共需要几页？
SELECT CEIL(25 / 10) AS 总页数;   -- 结果：3（不能只有2页，第3页装剩余5条）

-- 动态计算学生总记录数需要的页数（每页 3 条）
SELECT CEIL(COUNT(*) / 3) AS 总页数 FROM students;
```

## 03.3  FLOOR —— 向下取整

`FLOOR` 总是返回**小于或等于**该数值的最大整数，也就是直接截掉小数部分，永远向下取整

```sql
-- 基本语法
SELECT FLOOR(数值);

-- 实际应用：计算折扣后价格并向下取整（对消费者有利）
SELECT
    product,
    unit_price,
    FLOOR(unit_price * 0.8) AS 八折后向下取整价格
FROM orders;
```

## 03.4 TRUNCATE —— 截断

`TRUNCATE` 直接截断到指定小数位，**不进行四舍五入**，多余的小数位直接丢弃。这与 `ROUND` 的区别在于：`ROUND` 会根据后续数字决定是否进位，而 `TRUNCATE` 永远直接截断

```sql
-- 基本语法
SELECT TRUNCATE(数值, 小数位数);

-- 实际应用：财务场景中，金额截断到分（不四舍五入，避免多收钱）
SELECT
    product,
    unit_price,
    TRUNCATE(unit_price * 0.75, 2) AS 七五折截断价格,
    ROUND(unit_price * 0.75, 2)    AS 七五折四舍五入价格
FROM orders;
```

## 03.5 ABS —— 绝对值

`ABS` 返回数值的绝对值，即去掉符号后的值，负数变正数，正数和零不变

```sql
-- 基本语法
SELECT ABS(数值);

-- 实际应用：计算每个学生的成绩与平均分的偏差（不管高于还是低于）
SELECT
    name,
    score,
    ROUND(AVG(score) OVER(), 2)             AS 全体平均分,
    ABS(score - AVG(score) OVER())          AS 与平均分的偏差
FROM students
ORDER BY 与平均分的偏差 DESC;
```

## 03.6 RAND —— 随机浮点数

```sql
-- 基本语法
SELECT RAND();              -- 生成 [0, 1) 范围内的随机浮点数
SELECT RAND(种子值);         -- 使用固定种子，每次结果相同（可重现）

```

**生成指定范围内的随机整数**

生成 [m,n] 范围内的随机整数，公式为：

$\text{FLOOR}(\text{RAND()} \times (n - m + 1)) + m$

```sql
-- 生成 1 到 100 之间的随机整数
SELECT FLOOR(RAND() * 100) + 1 AS 随机整数;

-- 生成 60 到 100 之间的随机整数（模拟随机成绩）
SELECT FLOOR(RAND() * 41) + 60 AS 随机成绩;

-- 生成 1 到 6 之间的随机整数（模拟掷骰子）
SELECT FLOOR(RAND() * 6) + 1 AS 骰子点数;

-- 随机抽取大约 30% 的学生数据（用于数据抽样分析）
SELECT name, score
FROM students
WHERE RAND() < 0.3;
```

# 04 日期函数

## 04.1 NOW函数

`NOW()` 返回当前的**日期和时间**，格式为 `YYYY-MM-DD HH:MM:SS`。它返回的是语句**开始执行时**的时间，在同一条语句中多次调用 `NOW()`，返回值完全相同

```sql
-- 基本语法
SELECT NOW();                   -- 结果：2026-03-04 14:30:25
SELECT NOW() + 0;               -- 结果：20260304143025（转为数值）

-- 实际应用：插入数据时记录当前时间
INSERT INTO orders (student_id, product, quantity, unit_price, is_paid, created_at)
VALUES (1, 'Python教材', 1, 59.00, 0, NOW());

-- 实际应用：查询今天创建的订单
SELECT * FROM orders
WHERE created_at >= NOW() - INTERVAL 24 HOUR;
```

## 04.2 CURDATE / CURRENT_DATE 函数

只返回当前**日期**部分，格式为 `YYYY-MM-DD`，不包含时间

```sql
-- 基本语法
SELECT CURDATE();
SELECT CURRENT_DATE();    -- 与 CURDATE() 完全等价
SELECT CURRENT_DATE;      -- 省略括号也合法

-- 实际应用：查询今天创建的记录
SELECT * FROM orders
WHERE DATE(created_at) = CURDATE();

-- 实际应用：计算学生从今天起还有几天生日（假设有 birthday 列）
SELECT name, birthday,
    DATEDIFF(
        DATE(CONCAT(YEAR(CURDATE()), '-', MONTH(birthday), '-', DAY(birthday))),
        CURDATE()
    ) AS 距今天天数
FROM students;
```

## 04.3 CURTIME / CURRENT_TIME函数

只返回当前**时间**部分，格式为 `HH:MM:SS`，不包含日期

```sql
-- 实际应用：判断当前是否在营业时间内（09:00 ~ 18:00）
SELECT
    CASE
        WHEN CURTIME() BETWEEN '09:00:00' AND '18:00:00' THEN '营业中'
        ELSE '已打烊'
    END AS 营业状态;
```

## 04.4 提取日期列里的年月日

```sql
year(date)
month(date)
day(date)
```

## 04.5 DATE_ADD()

```mysql
-- 基本语法
DATE_ADD(日期, INTERVAL 数值 单位)

-- 加一个月
SELECT DATE_ADD('2026-03-09', INTERVAL 1 MONTH);

-- 加2小时
SELECT DATE_ADD('2026-03-09 10:00:00', INTERVAL 2 HOUR);
```

> * DATE_SUB() 日期减法函数，与加法的写法一致

## 04.6 DATEDIFF()

返回两个日期之间相差的**天数**,==是用参数1减去参数2的==

```sql
SELECT DATEDIFF('2026-03-09', '2026-03-01');
```

# 05 流程控制函数

## 05.1 `IF()` —— 简单的二元分支

```sql
-- 基本语法
IF(条件表达式, 表达式为真时的返回值, 表达式为假时的返回值)

-- 将 is_active 的 1 和 0 转换为直观的文字
SELECT 
    name, 
    is_active, 
    IF(is_active = 1, '已激活', '未激活') AS 账号状态 
FROM students;
```

**局限性**：`IF()` 函数只能处理**非黑即白**的二元逻辑。如果遇到多种情况（如优秀、良好、及格、不及格），虽然可以嵌套 `IF(..., IF(...), ...)`，但代码会变得极其难以阅读。面对多分支，请使用 `CASE WHEN`

## 05.2 IFNULL 与 COALESCE函数

在 SQL 中，`NULL` 是一个极其特殊的存在。任何包含 `NULL` 的算术运算（如 `10 + NULL`）或字符串拼接（如 `CONCAT('A', NULL)`），结果都会变成 `NULL`。为了防止 `NULL` 破坏业务逻辑，必须对其进行兜底处理

```sql
-- 基本语法
IFNULL(表达式1, 表达式2);
-- 如果“表达式1”不为 NULL，则返回“表达式1”的值；如果为 NULL，则返回“表达式2”的值。

-- 场景：展示邮箱时，如果没有填写则显示“暂无”
SELECT 
    name, 
    IFNULL(email, '暂无邮箱') AS 联系方式
FROM students;

-- 场景：计算订单实际支付金额时，如果 discount（折扣）为 NULL，则当作 0 计算
SELECT 
    product, 
    unit_price, 
    quantity, 
    (unit_price * quantity) - IFNULL(discount, 0) AS 实际支付金额
FROM orders;
```



**COALESCE()是标准 SQL**，在 MySQL、Oracle、SQL Server、PostgreSQL 中通用。**支持多个参数**，比 `IFNULL`（只能传两个参数）更灵活

```sql
-- 基本语法
COALESCE(值1, 值2, 值3, ...)
-- COALESCE 会从左到右依次检查参数，返回第一个不为 NULL 的值。如果所有参数都为 NULL，则返回 NULL。

-- 场景：获取用户的联系方式，优先取手机号，没有手机号取邮箱，都没有则返回“无联系方式”
SELECT 
    name, 
    COALESCE(phone, email, '无联系方式') AS 最终联系方式
FROM students;
```

## 05.3 CASE WHEN函数

```sql
-- 用法一
CASE 字段或表达式
    WHEN 值1 THEN 结果1
    WHEN 值2 THEN 结果2
    ...
    [ELSE 默认结果]
END

-- 场景：将性别枚举值翻译为英文
SELECT 
    name, 
    gender,
    CASE gender
        WHEN '男' THEN'Male'
        WHEN '女' THEN'Female'
        ELSE'Unknown'
    END AS gender_en
FROM students;

-- 用法二
CASE 
    WHEN 条件1 THEN 结果1
    WHEN 条件2 THEN 结果2
    ...
    [ELSE 默认结果]
END
-- 执行顺序：从上到下依次判断，只要命中一个 WHEN，就会返回对应的 THEN 并结束匹配（跳过后面的所有 WHEN）
```

```sql
SELECT 
    name, 
    score,
    CASE 
        WHEN score >= 90 THEN '优秀'
        WHEN score >= 80 THEN '良好'
        WHEN score >= 60 THEN '及格'
        ELSE '不及格'
    END AS 成绩评级
FROM students;

-- 场景：判断用户是否为“高价值且活跃用户”
SELECT 
    name,
    score,
    is_active,
    CASE 
        WHEN is_active = 1 AND score >= 85 THEN '优质活跃用户'
        WHEN is_active = 1 AND score < 85 THEN '普通活跃用户'
        WHEN is_active = 0 THEN '沉睡用户'
        ELSE '状态异常'
    END AS 用户分层
FROM students;
```



利用 CASE WHEN 实现“行转列”（数据透视）

![image-20260305123318952](F:\note\MySQL\day04_函数.assets\image-20260305123318952.png)

```sql
SELECT 
    '总人数统计' AS 指标名称,
    SUM(CASE WHEN gender = '男' THEN 1 ELSE 0 END) AS 男生人数,
    SUM(CASE WHEN gender = '女' THEN 1 ELSE 0 END) AS 女生人数,
    SUM(CASE WHEN gender = '保密' THEN 1 ELSE 0 END) AS 保密人数,
    COUNT(*) AS 总计
FROM students;
```

![image-20260305123343088](F:\note\MySQL\day04_函数.assets\image-20260305123343088.png)

