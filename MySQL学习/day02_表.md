# 01 创建表

## 01.1 基本语法

```mysql
CREATE TABLE 表名 (
    列名1 数据类型 [约束条件],
    列名2 数据类型 [约束条件],
    ...
    [表级约束]
);
```

示例：

```mysql
CREATE TABLE students (
    id        INT           NOT NULL AUTO_INCREMENT,
    name      VARCHAR(50)   NOT NULL,
    age       TINYINT       UNSIGNED,
    email     VARCHAR(100)  UNIQUE,
    score     DECIMAL(5, 2) DEFAULT 0.00,
    gender    ENUM('男', '女', '保密') DEFAULT '保密',
    create_at DATETIME      DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);
```

## 01.2 Mysql数据类型（列类型）

### 01.2.1 数值类型

![image-20260302140904220](F:\note\数据库\MySQL\day02_表.assets\image-20260302140904220.png)

![image-20260302141045997](F:\note\数据库\MySQL\day02_表.assets\image-20260302141045997.png)

![image-20260302141126752](F:\note\数据库\MySQL\day02_表.assets\image-20260302141126752.png)

### 01.2.2 字符串类型

![image-20260302141312100](F:\note\数据库\MySQL\day02_表.assets\image-20260302141312100.png)

![image-20260302141415893](F:\note\数据库\MySQL\day02_表.assets\image-20260302141415893.png)

### 01.2.3 二进制数据类型

![image-20260302141505938](F:\note\数据库\MySQL\day02_表.assets\image-20260302141505938.png)

### 01.2.4 日期/时间类型

![image-20260302141708871](F:\note\数据库\MySQL\day02_表.assets\image-20260302141708871.png)

# 02 修改表

## 02.1 修改列

> * 操作前要谨慎些，该备份备份，该检查检查

```sql
-- 1. 先用 DESC 查看当前表结构，再动手修改
DESC students;

-- 2. 用 SHOW CREATE TABLE 查看完整建表语句（含约束、索引名）
SHOW CREATE TABLE students;

-- 3. 重要数据修改前先备份
CREATE TABLE students_bak AS SELECT * FROM students;
```

### 02.1.1 添加列

```sql
-- 基本语法
ALTER TABLE 表名 ADD COLUMN 列名 数据类型 [约束];

-- 示例：在 students 表末尾添加一个手机号列
ALTER TABLE students ADD COLUMN phone VARCHAR(20);

-- 添加到最前面（FIRST）
ALTER TABLE students ADD COLUMN uid INT UNSIGNED FIRST;

-- 添加到指定列之后（AFTER）
ALTER TABLE students ADD COLUMN nickname VARCHAR(50) AFTER name;
```

### 02.1.2 删除列

```sql
-- 基本语法
ALTER TABLE 表名 DROP COLUMN 列名;

-- 示例：删除 phone 列
ALTER TABLE students DROP COLUMN phone;
```

### 02.1.3 修改列的数据类型

> * MODIFY 会**覆盖原有的所有约束**，即使只想改类型，也要把原来的约束一起写上，否则会丢失

```sql
-- 基本语法
ALTER TABLE 表名 MODIFY COLUMN 列名 新数据类型 [新约束];

-- 示例：把 age 从 TINYINT 改为 SMALLINT
ALTER TABLE students MODIFY COLUMN age SMALLINT UNSIGNED;

-- 示例：给 name 加上 NOT NULL 约束
ALTER TABLE students MODIFY COLUMN name VARCHAR(50) NOT NULL;

-- 示例：修改默认值
ALTER TABLE students MODIFY COLUMN score DECIMAL(5,2) DEFAULT 60.00;
```

### 02.1.4 重命名列（同时可改类型）

```sql
-- 基本语法
ALTER TABLE 表名 CHANGE COLUMN 旧列名 新列名 数据类型 [约束];

-- 示例：把 name 重命名为 full_name，类型保持不变
ALTER TABLE students CHANGE COLUMN name full_name VARCHAR(50) NOT NULL;

-- 示例：把 age 重命名为 user_age，同时改类型
ALTER TABLE students CHANGE COLUMN age user_age SMALLINT UNSIGNED;
```

### 02.1.5 修改列的位置

```sql
-- 把 email 列移到最前面
ALTER TABLE students MODIFY COLUMN email VARCHAR(100) FIRST;

-- 把 email 列移到 name 列之后
ALTER TABLE students MODIFY COLUMN email VARCHAR(100) AFTER name;
```

### 02.1.6 重命名表

```sql
-- 方式一：ALTER TABLE（标准写法）
ALTER TABLE students RENAME TO stu_info;

-- 方式二：RENAME TABLE（更简洁，支持同时改多张表）
RENAME TABLE students TO stu_info;
RENAME TABLE t1 TO t1_bak, t2 TO t2_bak;   -- 同时改多张表
```



==多个修改用逗号分隔，**效率远高于多条单独的 ALTER TABLE**（只需重建一次表结构）==

```sql
ALTER TABLE students
    ADD COLUMN phone    VARCHAR(20)  AFTER email,
    ADD COLUMN address  VARCHAR(200),
    MODIFY COLUMN score DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    ADD INDEX idx_phone (phone);
```

# 03 删除表

> * 以后不再用这张表 → `DROP`；表还要用只是清空数据 → `TRUNCATE`

```sql
-- 删除表（表和数据全删）
DROP TABLE 表名;
DROP TABLE IF EXISTS 表名;               -- 安全删除，不存在不报错
DROP TABLE IF EXISTS t1, t2, t3;        -- 同时删除多张表

-- 清空表数据（保留表结构）
TRUNCATE TABLE 表名;

-- 备份表（数据完整，但无主键索引）
CREATE TABLE 备份表名 AS SELECT * FROM 原表名;

-- 备份部分数据
CREATE TABLE 备份表名 AS SELECT * FROM 原表名 WHERE 条件;

-- 复制完整结构（无数据）
CREATE TABLE 备份表名 LIKE 原表名;

-- 最完整备份（结构 + 数据 + 主键索引）
CREATE TABLE 备份表名 LIKE 原表名;
INSERT INTO 备份表名 SELECT * FROM 原表名;

-- 从备份恢复数据
TRUNCATE TABLE 原表名;
INSERT INTO 原表名 SELECT * FROM 备份表名;
```











































