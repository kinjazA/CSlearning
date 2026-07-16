# 01 SQL数据库初入

> * 多条sql语句要用分号来间隔
> * sql语句不区分大小写
> * sql语句中的空格会被忽略，可以写成一行，也可以写成多行

## 01.1 命令行操作MySQL

> * **mysql服务启动**：要在管理员权限下（这里要在打开cmd前选以管理员身份运行）进入mysql的bin目录`cd /D D:\APP\mysql\mysql-5.7.19-winx64\bin`，输入`net start mysql`，停止服务就输入`net stop mysql`
> * root用户密码设的是空，直接回车进入
> * 修改了root用户的密码`update user set authentication_string=password('lyh')where user='root' and Host='localhost';`刷新权限后生效`flush privileges;`
> * 使用命令行连接到MySQL`mysql -h主机IP -P端口 -u 用 户名 -plyh`，如果不写-h -P就是默认的——`mysql -u root -plyh`

## 01.2 Navicat图形化MySQL管理软件

![image-20250711232839015](C:\Users\laiyouhua\AppData\Roaming\Typora\typora-user-images\image-20250711232839015.png)

## 01.3 MySQL三层结构

![image-20250711235215546](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250711235215546.png)

![image-20250711235232223](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250711235232223.png)

## 01.4 MySQL创建数据库

### 01.4.1 图形化软件操作

![image-20250712001200520](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250712001200520.png)

### 01.4.2 指令创建数据库

> * **CREATE DATABASE**
>     关键字，用于在 MySQL 中新建一个空的数据库（Schema）
> * **IF NOT EXISTS**
>     可选项。
>    - 含义：只有当数据库名在当前实例中不存在时才执行创建。
>    - 优势：避免重复创建时抛出错误，尤其适合在脚本里反复执行或部署过程中使用。
> * **`lyh_db02`**
>     数据库名：
>    - 使用反引号（``）可以防止名称与关键字冲突，也可包含特殊字符。
>    - 最好遵循“小写字母＋下划线”命名规范，如 `project_data`，提高可读性。
> * **DEFAULT CHARACTER SET utf8mb4**
>     设置字符集为 `utf8mb4`：
>    - 支持完整的 Unicode 编码，包括 Emoji、𤭢 等。
>    - 如果不指定，则使用服务器或实例级默认字符集.
> * **OLLATE utf8mb4_general_ci**
>     设置默认校对规则（排序和比较规则）：
>    - `utf8mb4_general_ci`：大小写不敏感（ci = case-insensitive），且对多数语言通用。
>    - 也可选用 `utf8mb4_unicode_ci`（更严格的 Unicode 比较，但性能略低）或区域性校对如 `utf8mb4_zh_0900_as_cs`（区分大小写、声调等）。

```sql
CREATE DATABASE IF NOT EXISTS `lyh_db02`
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_general_ci;
```

![image-20250712001809897](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250712001809897.png)

![image-20250712002216187](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250712002216187.png)

### 01.4.3 查看、删除数据库

```mysql
#列出当前 MySQL 实例中所有可见的数据库,登录的用户权限决定了能看到哪些数据库。
SHOW DATABASES;

#删除数据库(慎重)
DROP DATABASE [IF EXISTS] 数据库名;

```

> * ### 参数解释：
>
>    - `DROP DATABASE`：删除整个数据库及其中的所有表和数据。
>    - `IF EXISTS`：可选参数，表示只有当该数据库存在时才执行删除，否则不会报错。

![image-20250712100557419](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250712100557419.png)

### 01.4.4 备份、恢复数据库

> * mysqldump 应该在操作系统命令行中执行（**不是**在 mysql 客户端）！

![image-20250712101739562](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250712101739562.png)

![image-20250712103501715](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250712103501715.png)

> * 若只想备份数据库里的某几个表

```powershell
mysqldump -u 用户名 -p 数据库名 表1 表2 表3 > 备份文件.sql
```

> * 当已经进入了 MySQL 命令行客户端（`mysql>` 提示符），可以使用 `source` 来导入 `.sql` 文件中的语句。
> * `source` 是 **MySQL 客户端内部命令**，用于在 `mysql>` 交互环境中**执行一个 SQL 脚本文件（.sql）**。相当于你把文件里的 SQL 语句，一条一条粘贴进来执行。

![image-20250712105326214](F:\note\数据库\MySQL\day01_初入SQL.assets\image-20250712105326214.png)



# 02 SQL通用语法

> * 可以单行或多行书写，一句命令以分号结尾
> * 可以使用空格或是缩进来增强语句的可读性
> * MySQL的SQL语句不区分大小写，建议关键字使用大写

# 03 SQL语句的分类

**DDL（Data Definition Language）数据定义语言**

**DML（Data Manipulation Language）数据操作语言**

**DQL（Data Query Language）数据查询语言**

**DCL（Data Control Language）数据控制语言**



