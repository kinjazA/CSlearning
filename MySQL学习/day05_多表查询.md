# 01 完全限定列名

## 01.1 应用范围

主要是为了解决 **列名冲突（ambiguous column）**，很多表可能都有相同列名

```sql
-- 写法1
-- 表名.列名
SELECT orders.id, users.name
FROM orders
JOIN users
ON orders.user_id = users.id;

-- 写法2
-- 表别名.列名
SELECT o.id, u.name
FROM orders o
JOIN users u
ON o.user_id = u.id;
```

![image-20260315084735728](day05_%E5%A4%9A%E8%A1%A8%E6%9F%A5%E8%AF%A2.assets/image-20260315084735728.png)

## 01.2 反引号

反引号（\` \`）在 MySQL 中叫：**标识符引用符（identifier quote）**。作用是：**把里面的内容当作“列名 / 表名 / 别名”，而不是关键字或表达式**，避免名字和关键字冲突



1️⃣ 别名 / 列名是关键字

```sql
SELECT score,
       DENSE_RANK() OVER (ORDER BY score DESC) AS rank   -- ❌ 可能报错
FROM Scores;
```

因为 `rank` 可能被解析器当成关键字，正确写法：

```sql
SELECT score,
       DENSE_RANK() OVER (ORDER BY score DESC) AS `rank` -- ✅
FROM Scores;
```



 2️⃣ 名字里有空格

```sql
SELECT score AS `my score`
FROM Scores;
```

👉 不加反引号会直接报错



3️⃣ 名字里有特殊字符

比如：

```sql
SELECT score AS `score%`
SELECT score AS `user-name`
SELECT score AS `@rank`
```

这些都必须用反引号



 4️⃣ 表名 / 列名本身是关键字

```sql
SELECT `select`, `from`
FROM `order`;
```

如果表/字段叫这些名字（不推荐，但现实中会遇到）



 5️⃣ 区分大小写（某些系统）

在 Linux + MySQL 下：

```sql
SELECT * FROM `User`;
```

和

```sql
SELECT * FROM user;
```

可能是不同表（取决于配置）

# 02 子查询

==就是把一条select子句的返回结果作为另一条select子句的where或from等里面==

当用于 `WHERE` 子句时，根据不同的运算符，子查询可以返回单行单列、多行单列、单行多列数据。子查询就是要返回能够作为 WHERE 子句查询条件的值

当用于 `FROM` 子句时，一般返回多行多列数据，相当于返回一张临时表，这样才符合 `FROM` 后面是表的规则。这种做法能够实现多表联合查询



## 02.1 标量子查询（返回单行单列）

这是最简单的子查询，它就像一个**单一的值**。**适用运算符**：`=`、`>`、`<`、`>=`、`<=`、`<>`

```sql
-- 场景：查询工资高于公司平均工资的员工。
SELECT name, salary 
FROM employees 
WHERE salary > (
    SELECT AVG(salary) FROM employees  -- 子查询返回一个确定的数值（如 7500）
);
```

## 02.2 列子查询（返回多行单列）

子查询返回的是**一列数据的集合**，相当于一个列表（List）。**适用运算符**：`IN`、`NOT IN`、`ANY`、`ALL`

```sql
-- 场景：查询“研发部”和“市场部”的所有员工。
SELECT name, dept_id 
FROM employees 
WHERE dept_id IN (
    SELECT id FROM departments WHERE name IN ('研发部', '市场部') -- 返回 (1, 2)
);
```

## 02.3 行子查询（返回单行多列）

子查询返回一条完整的记录（多个字段的值）。 **适用运算符**：`=`, `<>`, `IN`

```sql
-- 场景：查询与 Bob 部门相同且工资也相同的其他员工。
SELECT name, dept_id, salary 
FROM employees 
WHERE (dept_id, salary) = (
    SELECT dept_id, salary FROM employees WHERE name = 'Bob'
) AND name != 'Bob';
```

## 02.4 表子查询（返回多行多列）

子查询返回的结果就像一张**虚拟的临时表**。通常放在 `FROM` 子句中使用。

==注意：**FROM 中的子查询必须起别名**。==

```sql
-- 场景：先统计每个部门的平均工资，再从中找出平均工资大于 8000 的部门。
SELECT dept_id, avg_salary 
FROM (
    -- 将这个子查询的结果当作一张名为 temp_table 的表
    SELECT dept_id, AVG(salary) AS avg_salary 
    FROM employees 
    GROUP BY dept_id
) AS temp_table 
WHERE avg_salary > 8000;
```

## 02.5 常用关键字

### 02.5.1 集合比较类

#### 1. IN/NOT IN

判断某个值是否在子查询返回的结果集合中 / 判断某个值**不在**子查询返回的集合中

```sql
-- 基本语法
expr IN (subquery)
expr NOT IN (subquery)

-- 员工的 dept_id 只要出现在子查询返回的部门 id 集合中，就满足条件
SELECT *
FROM employees
WHERE dept_id IN (
    SELECT id
    FROM departments
    WHERE location = 'Beijing'
);

-- 查询不属于北京部门的员工。
SELECT *
FROM employees
WHERE dept_id NOT IN (
    SELECT id
    FROM departments
    WHERE location = 'Beijing'
);
```

子查询结果中如果有 `NULL`，`NOT IN` 很容易出问题

#### 2. ANY/SOME

与子查询结果中的**任意一个值**比较，只要满足一个就成立

```css
x > ANY(S) 等价于 x > MIN(S)
x < ANY(S) 等价于 x < MAX(S)
```

```sql
-- 工资只要高于 10 号部门中任意一个人的工资即可
SELECT *
FROM employees
WHERE salary > ANY (
    SELECT salary
    FROM employees
    WHERE dept_id = 10
);

-- 和 > ANY (...) 完全一样
SELECT *
FROM employees
WHERE salary > SOME (
    SELECT salary
    FROM employees
    WHERE dept_id = 10
);
```

####  3. ALL

与子查询结果中的**所有值**比较，必须全部满足才成立。

```css
x > ALL(S) 等价于 x > MAX(S)
x < ALL(S) 等价于 x < MIN(S)
```

```sql
-- 工资必须高于 10 号部门所有员工
SELECT *
FROM employees
WHERE salary > ALL (
    SELECT salary
    FROM employees
    WHERE dept_id = 10
);
```

### 02.5.2 存在性判断类

#### 1. EXISTS/NOT EXISTS

判断子查询**是否至少返回一行记录**。 / 判断子查询**是否没有返回记录**



假设有两张表：

1. **`customers`（顾客表）**：记录了所有注册的用户。
2. **`orders`（订单表）**：记录了用户下过的订单

```sql
-- 业务需求：找出所有“下过单的活跃顾客”
SELECT name 
FROM customers c
WHERE EXISTS (
    SELECT 1 
    FROM orders o 
    WHERE o.customer_id = c.id
);
```

1. **第一步：拿起“张三 (id=1)”**
   - 探测器（子查询）去订单表里找：`SELECT 1 FROM orders WHERE customer_id = 1`
   - 探测器滴滴滴响了！找到了订单 101 和 102
   - `EXISTS` 报告：**TRUE（存在）**
   - **结果**：把“张三”放进最终的输出列表里
2. **第二步：拿起“李四 (id=2)”**
   - 探测器去订单表里找：`SELECT 1 FROM orders WHERE customer_id = 2`
   - 探测器响了！找到了订单 103
   - `EXISTS` 报告：**TRUE（存在）**
   - **结果**：把“李四”放进最终的输出列表里
3. **第三步：拿起“王五 (id=3)”**
   - 探测器去订单表里找：`SELECT 1 FROM orders WHERE customer_id = 3`
   - 探测器没反应，订单表里根本没有 `customer_id = 3` 的记录
   - `EXISTS` 报告：**FALSE（不存在）**
   - **结果**：把“王五”无情抛弃，不输出
4. **第四步：拿起“赵六 (id=4)”**
   - 同理，找不到订单，`EXISTS` 返回 **FALSE**。抛弃

**最终输出结果**：张三，李四

> **💡 关键点解惑：为什么里面写的是 `SELECT 1`？**
>  因为 `EXISTS` 根本不在乎查出来了什么具体数据（金额是多少、订单号是多少它都不管），它**只在乎“有没有”这行记录**。所以写 `SELECT 1`（随便返回个常数 1）是最高效的写法，告诉数据库“只要找到记录，回传个 1 证明找到了就行”

```sql
-- 业务需求：找出所有“只注册但从未下过单的白嫖顾客”
SELECT name 
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 
    FROM orders o 
    WHERE o.customer_id = c.id
);
```

==**`EXISTS`**：拿着外面的数据，去里面的表里找。**找到了，就保留外面这行数据**；找不到，就扔掉。==

==**`NOT EXISTS`**：拿着外面的数据，去里面的表里找。**找不到，才保留外面这行数据**；找到了，反而扔掉。==

# 03 连结表

在关系型数据库中，为了避免数据冗余，通常把不同实体的数据分开存（比如人存一张表，订单存另一张表）。而 **JOIN（连结）** 的作用，就是像拉链一样，把这些分散的表重新“缝合”在一起，能一次性查出完整的信息



![image-20260306144252959](F:\note\MySQL\day05_多表查询.assets\image-20260306144252959.png)

## 03.1 INNER JOIN（内连结）

==有时候也会省略不写inner==

```sql 
-- 业务需求：只关心“成功匹配”的数据。既要有顾客信息，又要有订单信息
SELECT 
    c.name AS 顾客姓名, 
    o.order_id AS 订单号, 
    o.product AS 商品
FROM customers c
INNER JOIN orders o         -- 核心关键字
ON c.id = o.customer_id;    -- 连结条件：两边的 ID 必须对得上
```
## 03.2 LEFT JOIN（左连结）

```sql
-- 业务需求：我要看所有顾客的消费情况，哪怕他从来没买过东西，名字也得给我列出来。
SELECT 
    c.name AS 顾客姓名, 
    o.order_id AS 订单号, 
    o.product AS 商品
FROM customers c            -- 左表：顾客表（老大）
LEFT JOIN orders o          -- 核心关键字
ON c.id = o.customer_id;
```

右连结换一下表位置，全写成左连结就好了，不用重复看

## 03.3 外连结









## 03.4 自连结





















