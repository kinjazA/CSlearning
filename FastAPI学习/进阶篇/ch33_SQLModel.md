前面所有章节的数据都存在 Python 字典里——程序一重启数据就没了。从这一节开始，数据终于要存进**真正的数据库**了



FastAPI 不绑定任何数据库，想用什么都行。但官方推荐 **SQLModel**——它是 FastAPI 作者自己写的库，把 **SQLAlchemy**（Python 最流行的数据库工具）和 **Pydantic**（你已经很熟的数据验证库）合二为一



**一个类既是数据库表的定义，又是请求/响应的数据模型。** 不用写两套代码了这一节的内容量很大，核心就三件事：

1.怎么用 SQLModel 定义表、连接数据库、做 CRUD

2.怎么用 `yield` 依赖项管理数据库会话（每个请求一个会话）

3.怎么用多模型继承体系保护敏感数据（创建模型、表模型、公开模型、更新模型）

---

## 1. 先搞懂几个概念

### 1.1 ORM 是什么

ORM（Object-Relational Mapping，对象关系映射）让你用 Python 类和对象来操作数据库，不用手写 SQL

==ORM将数据库中的表映射为编程语言中的类，每一行数据对应一个对象，标的列则对应对象的属性==

```python
# 不用 ORM → 手写 SQL
cursor.execute("INSERT INTO heroes (name, age) VALUES ('Spider-Boy', 18)")
cursor.execute("SELECT * FROM heroes WHERE age > 16")

# 用 ORM → 写 Python 代码
hero = Hero(name="Spider-Boy", age=18)
session.add(hero)                          # 等价于 INSERT
heroes = session.exec(select(Hero).where(Hero.age > 16)).all()  # 等价于 SELECT
```

### 1.2 SQLModel = SQLAlchemy + Pydantic

```css
SQLAlchemy：Python 最流行的数据库 ORM，功能强大但写起来繁琐
Pydantic：  你已经学过的数据验证库，FastAPI 的核心
SQLModel：  把两者合并，一个类同时搞定数据库表定义和数据验证

之前你要写两个类：
  class HeroTable(SQLAlchemy Base)    → 定义数据库表
  class HeroSchema(Pydantic BaseModel) → 定义 API 数据模型

现在只写一个类：
  class Hero(SQLModel, table=True)     → 既是数据库表，又是数据模型
```

### 1.3 这一节用 SQLite

SQLite 是一个文件数据库——整个数据库就是一个 `.db` 文件，不需要安装数据库服务器，Python 自带支持。开发和学习阶段用 SQLite，生产环境换 PostgreSQL（只需改一行连接字符串）

---

## 2. 从最简单的版本开始——单模型 CRUD

### 2.1 完整代码（先看全貌）

```python
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select


# ==================== 模型 ====================

class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
    secret_name: str


# ==================== 数据库连接 ====================

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


# ==================== 依赖项：数据库会话 ====================

def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


# ==================== FastAPI 应用 ====================

app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# ==================== CRUD 接口 ====================

@app.post("/heroes/")
def create_hero(hero: Hero, session: SessionDep) -> Hero:
    session.add(hero)
    session.commit()
    session.refresh(hero)
    return hero


@app.get("/heroes/")
def read_heroes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Hero]:
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes


@app.get("/heroes/{hero_id}")
def read_hero(hero_id: int, session: SessionDep) -> Hero:
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero


@app.delete("/heroes/{hero_id}")
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(hero)
    session.commit()
    return {"ok": True}
```

下面逐块拆解

### 2.2 定义模型（数据库表）

```python
class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True) # Field用于Pydantic模型加参数约束
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
    secret_name: str
```

逐字段解释：

```css
class Hero(SQLModel, table=True):
                     ^^^^^^^^^^
                     table=True → 这个类对应数据库里的一张表
                     没有 table=True 的 SQLModel 类就是普通的 Pydantic 数据模型

id: int | None = Field(default=None, primary_key=True)
    ^^^^^^^^^^                       ^^^^^^^^^^^^^^^^
    类型是 int | None                 这是主键

    为什么是 int | None 而不是 int？
    → 创建英雄时还没有 id（id=None），数据库会自动生成
    → 从数据库读出来之后 id 就有值了（是 int）
    → 数据库里这列定义为 INTEGER NOT NULL（SQLModel 知道主键不能为 NULL）

name: str = Field(index=True)
                  ^^^^^^^^^^^
                  index=True → 给这列创建数据库索引
                  索引加速按该列的查询（WHERE name = '...'）

age: int | None = Field(default=None, index=True)
    ^^^^^^^^^^
    int | None → 年龄是可选的，可以不填（数据库里允许 NULL）

secret_name: str
    → 普通的必填字段，没有特殊配置
```

这个类在数据库里创建的表等价于这条 SQL：

```sql
CREATE TABLE hero (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    name VARCHAR NOT NULL,
    age INTEGER,
    secret_name VARCHAR NOT NULL
);
CREATE INDEX ix_hero_name ON hero (name);
CREATE INDEX ix_hero_age ON hero (age);
```

### 2.3 创建引擎（Engine）

```python
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)
```

逐行解释：

sqlit:///./database.db，拆看看，qlite表示数据库的类型，://是固定符合，/./database.db表示文件路径。因为通用的URL一般是`协议://主机/路径`，但SQLite没有主机，所以主机位置是空的，`sqlite:///./app.db`

```css
sqlite_url = "sqlite:///database.db"
              ^^^^^^^^^^
              数据库连接字符串（URL 格式）
              sqlite:///  → 使用 SQLite，文件在当前目录
              database.db → 数据库文件名

              其他数据库的连接字符串：
              PostgreSQL：postgresql://user:password@localhost:5432/mydb
              MySQL：     mysql://user:password@localhost:3306/mydb

connect_args = {"check_same_thread": False}
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
              SQLite 特有的设置
              SQLite 默认不允许多线程访问同一个连接
              FastAPI 可能在不同线程处理同一个请求（比如依赖项和路径操作函数在不同线程）
              所以要关掉这个检查
              PostgreSQL/MySQL 不需要这个设置

engine = create_engine(sqlite_url, connect_args=connect_args)
              ^^^^^^^^^^^^^^^^^^
              Engine 是整个应用和数据库之间的"桥梁"
              整个应用只需要一个 Engine（全局单例）
              Engine 内部维护着数据库连接池
```

### 2.4 创建表

```python
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
```

`SQLModel.metadata.create_all(engine)` 会检查所有 `table=True` 的模型类，如果数据库里还没有对应的表就创建它。已经存在的表不会被修改或删除——这个方法是安全的，可以重复调用

### 2.5 数据库会话（Session）——用 yield 依赖项管理

```python
def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
```

这是本节**最重要的代码之一**。逐步拆解：

**Session 是什么？**

```css
Engine  → 管理数据库连接（全局一个）
Session → 管理一次"对话"（每个请求一个）

Session 的职责：
  1. 跟踪你在内存中创建/修改/删除的对象
  2. 在你调用 commit() 时把这些变更写入数据库
  3. 在你调用 rollback() 时撤销所有未提交的变更

类比：
  Engine 像银行大楼（全局一个）
  Session 像一次柜台服务（每个客户一次，办完走人）
```

**为什么用 yield 而不是 return？**

```python
# 如果用 return：
def get_session():
    session = Session(engine)
    return session
    # 问题：session 什么时候关闭？没人管它！
    # 数据库连接泄漏！

# 用 yield：
def get_session():
    with Session(engine) as session:
        yield session
    # yield 之后的代码在请求结束时执行
    # with 语句在退出时自动调用 session.close()
    # 数据库连接被正确释放 ✓
```

**yield 依赖项的执行流程**：

```css
请求进来
    │
    ▼
FastAPI 调用 get_session()
    │
    ▼
with Session(engine) as session:   ← 创建 Session
    yield session                   ← 把 session 交给路径操作函数
    │
    ▼
路径操作函数执行（使用 session 做数据库操作）
    │
    ▼
路径操作函数执行完毕
    │
    ▼
yield 之后的代码执行（这里是 with 的退出）
    │
    ▼
session.close() 被自动调用          ← 释放数据库连接
```

**SessionDep 类型别名**：

```python
SessionDep = Annotated[Session, Depends(get_session)]
```

和上一节学的一样——类型别名，多个接口复用，避免每次都写完整的 `Annotated[Session, Depends(get_session)]`

### 2.6 应用启动时创建表

```python
app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
```

`@app.on_event("startup")` 注册一个启动事件：FastAPI 应用启动时（第一个请求到来之前）自动执行。在这里创建数据库表，确保表结构就绪

生产环境通常不用这个方式，而是用数据库迁移工具（Alembic）在部署时管理表结构

### 2.7 CRUD 操作逐个讲解

**Create——创建**

```python
@app.post("/heroes/")
def create_hero(hero: Hero, session: SessionDep) -> Hero:
    session.add(hero)       # 第一步：把对象加入 Session（暂存在内存里）
    session.commit()        # 第二步：把 Session 里的变更写入数据库
    session.refresh(hero)   # 第三步：从数据库重新读取这条记录（拿到数据库生成的 id）
    return hero
```

为什么需要 `session.refresh(hero)`？

```css
创建前：hero.id = None（你没给 id，因为 id 由数据库自动生成）
commit 后：数据库生成了 id（比如 1），但 Python 里的 hero 对象还不知道
refresh 后：hero.id = 1（从数据库重新读取，同步了最新的值）

如果不 refresh，返回给客户端的 hero.id 就是 None——这显然不对。
```

三步口诀：**add → commit → refresh**

**Read（列表）——读取多条**

```python
@app.get("/heroes/")
def read_heroes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Hero]:
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes
```

拆解查询语句：

```python
select(Hero)                    # SELECT * FROM hero
    .offset(offset)             # OFFSET 0（跳过前 0 条 → 从第一条开始）
    .limit(limit)               # LIMIT 100（最多返回 100 条）

session.exec(...)               # 执行查询
    .all()                      # 把所有结果转成 Python 列表

等价的 SQL：
SELECT * FROM hero LIMIT 100 OFFSET 0
```

`limit` 限制了最大值为 100（`Query(le=100)`），防止客户端一次请求拉取全部数据

**Read（单条）——按 ID 读取**

```python
@app.get("/heroes/{hero_id}")
def read_hero(hero_id: int, session: SessionDep) -> Hero:
    hero = session.get(Hero, hero_id)    # 按主键查询
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero
```

`session.get(Hero, hero_id)` 是按主键查询的快捷方式，等价于：

```python
session.exec(select(Hero).where(Hero.id == hero_id)).first()
```

找不到就返回 `None`，我们手动返回 404

**Delete——删除**

```python
@app.delete("/heroes/{hero_id}")
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(hero)    # 标记为删除
    session.commit()        # 写入数据库
    return {"ok": True}
```

先查出来、再删除——两步。如果不先查，就不知道这条记录是否存在，也就无法返回 404

---

## 3. 单模型版本的两个安全问题

上面的代码能跑，但有两个严重问题：

### 3.1 问题一：客户端能指定 id

```python
# 创建接口接收 Hero 模型，Hero 模型有 id 字段
@app.post("/heroes/")
def create_hero(hero: Hero, session: SessionDep) -> Hero:
    ...
```

客户端可以发这样的请求体：

```json
{
    "id": 999,
    "name": "Evil Hero",
    "age": 30,
    "secret_name": "Hacker"
}
```

客户端指定了 `id: 999`，这条记录就会以 id=999 存入数据库——客户端能覆盖已有的记录。id 应该由数据库自动生成，不应该让客户端控制

### 3.2 问题二：secret_name 暴露了

```python
# 返回类型是 Hero，Hero 包含 secret_name
def create_hero(...) -> Hero:
    ...
    return hero    # 返回值里包含 secret_name！
```

所有接口都返回完整的 `Hero` 对象，`secret_name` 也被返回给客户端了。这个字段应该是秘密的，不能暴露在 API 响应里

**解决方案：用多个模型**

---

## 4. 多模型继承体系——这一节的核心设计

### 4.1 为什么需要多个模型

```css
场景                    需要的字段                         不需要的字段
───────────────────────────────────────────────────────────────────────
客户端创建英雄          name, age, secret_name             id（数据库生成）
数据库表结构            id, name, age, secret_name         （全都要）
API 返回给客户端        id, name, age                      secret_name（保密！）
客户端更新英雄          name?, age?, secret_name?           id（不能改）, 全可选
```

四个场景，需要四种不同的字段组合。一个类搞不定，需要四个类。但四个类之间有大量重复字段（name、age 几乎每个都有），怎么避免复制粘贴？**用继承**

### 4.2 四个模型的继承关系

```python
class HeroBase(SQLModel):  # 这里不用参数table，因为这个不需要创建表，只是把公共指标提取出来的基类
    """基类：所有模型共享的字段"""
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)


class Hero(HeroBase, table=True):
    """表模型：对应数据库表"""
    id: int | None = Field(default=None, primary_key=True)
    secret_name: str


class HeroPublic(HeroBase):
    """公开模型：API 返回给客户端的数据"""
    id: int


class HeroCreate(HeroBase):
    """创建模型：客户端创建英雄时发送的数据"""
    secret_name: str


class HeroUpdate(HeroBase):
    """更新模型：客户端更新英雄时发送的数据（全可选）"""
    name: str | None = None
    age: int | None = None
    secret_name: str | None = None
```

用一张图看清继承关系和每个模型有哪些字段：

```css
                    HeroBase
                   ┌─────────┐
                   │ name     │
                   │ age      │
                   └────┬────┘
            ┌───────────┼───────────┬───────────┐
            ▼           ▼           ▼           ▼
      Hero(table)    HeroPublic  HeroCreate  HeroUpdate
     ┌──────────┐   ┌──────────┐ ┌──────────┐ ┌──────────┐
     │ id       │   │ id       │ │secret_name│ │ name?    │
     │secret_name│  └──────────┘ └──────────┘ │ age?     │
     └──────────┘                             │secret_name?│
                                              └──────────┘

每个模型的完整字段：
  HeroBase:   name, age
  Hero:       id, name, age, secret_name          ← 数据库表（全字段）
  HeroPublic: id, name, age                       ← 返回给客户端（没有 secret_name）
  HeroCreate: name, age, secret_name              ← 客户端发来的（没有 id）
  HeroUpdate: name?, age?, secret_name?            ← 全部可选
```

### 4.3 逐个模型深入理解

**HeroBase——基类**

```python
class HeroBase(SQLModel):        # 注意：没有 table=True！
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
```

- 没有 `table=True` → 这只是一个数据模型（Pydantic 模型），不会在数据库里创建表
- 存放所有模型都共享的字段：`name` 和 `age`
- 其他模型继承它，就自动有了这两个字段，不用重复写

**Hero——表模型**

```python
class Hero(HeroBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    secret_name: str
```

- `table=True` → 对应数据库里的 `hero` 表
- 继承了 `name` 和 `age`，自己加了 `id` 和 `secret_name`
- `id: int | None` → 创建时 None，存入数据库后有值
- 这个类只在后端内部使用（和数据库交互），不直接暴露给客户端

**HeroPublic——公开模型（返回给客户端）**

```python
class HeroPublic(HeroBase):
    id: int                 # 注意：int 而不是 int | None
```

- 没有 `table=True` → 纯数据模型
- 没有 `secret_name` → 从数据库查出来返回给客户端时，secret_name 被过滤掉了
- `id: int` 不是 `int | None` → 和客户端的**契约**：API 返回的数据 id 一定有值、一定是整数

**HeroCreate——创建模型（客户端发来的数据）**

```python
class HeroCreate(HeroBase):
    secret_name: str
```

- 没有 `id` → 客户端不能指定 id，id 由数据库生成
- 有 `secret_name` → 创建时客户端要提供秘密名字
- 继承了 `name` 和 `age` → name 必填、age 可选

**HeroUpdate——更新模型（全可选）**

```python
class HeroUpdate(HeroBase):
    name: str | None = None          # 覆盖父类的 name: str → 变成可选
    age: int | None = None           # 本来就是可选，但默认值显式写 None
    secret_name: str | None = None   # 新增，可选
```

- 所有字段都是 `类型 | None = None` → 客户端只传要改的字段
- 配合 `model_dump(exclude_unset=True)` 使用（你在第二十二节学过这个技巧）
- 覆盖了父类的 `name` 字段（从必填变成可选）

### 4.4 为什么 HeroUpdate 要重新声明所有字段

你可能会问：`age` 在 `HeroBase` 里已经是 `int | None = None` 了，HeroUpdate 为什么还要重新写一遍？

```python
# HeroBase 里：
age: int | None = Field(default=None, index=True)

# HeroUpdate 里：
age: int | None = None
```

两个原因：

```css
1. name 在 HeroBase 里是必填的（name: str），但在 HeroUpdate 里必须变成可选的。
   所以 name 必须重新声明。

2. 既然 name 重新声明了，为了代码一致性和可读性，
   把 age 和 secret_name 也一起重新声明——一眼就能看清这个模型的所有字段都是可选的。
```

---

## 5. 多模型版本的 CRUD 接口

### 5.1 创建——接收 HeroCreate，返回 HeroPublic

```python
@app.post("/heroes/", response_model=HeroPublic)
def create_hero(hero: HeroCreate, session: SessionDep):
    db_hero = Hero.model_validate(hero)    # HeroCreate → Hero
    session.add(db_hero)
    session.commit()
    session.refresh(db_hero)
    return db_hero                          # 返回 Hero，但 response_model 过滤成 HeroPublic
```

逐步拆解：

```
第一步：客户端发来 JSON
  {"name": "Spider-Boy", "age": 18, "secret_name": "Pedro Parqueador"}

第二步：FastAPI 用 HeroCreate 模型解析
  → HeroCreate(name="Spider-Boy", age=18, secret_name="Pedro Parqueador")
  → 没有 id 字段 → 客户端无法指定 id ✓

第三步：Hero.model_validate(hero) 把 HeroCreate 转成 Hero
  → Hero(id=None, name="Spider-Boy", age=18, secret_name="Pedro Parqueador")
  → model_validate 从一个 Pydantic/SQLModel 对象创建另一个

第四步：add → commit → refresh
  → 数据库生成 id（假设是 1）
  → refresh 后：Hero(id=1, name="Spider-Boy", age=18, secret_name="Pedro Parqueador")

第五步：return db_hero
  → 返回的是 Hero 对象（包含 secret_name）
  → 但 response_model=HeroPublic 告诉 FastAPI：
    "用 HeroPublic 来序列化响应"
  → HeroPublic 没有 secret_name 字段
  → 最终客户端收到：{"id": 1, "name": "Spider-Boy", "age": 18}
  → secret_name 被过滤掉了 ✓
```

**为什么用 `response_model=HeroPublic` 而不是 `-> HeroPublic`？**

```python
# 如果用返回类型注解：
def create_hero(...) -> HeroPublic:
    ...
    return db_hero    # 返回的是 Hero 对象，不是 HeroPublic
    # 编辑器会报警：返回类型不匹配！

# 所以用 response_model 参数：
@app.post("/heroes/", response_model=HeroPublic)
def create_hero(...):
    ...
    return db_hero    # 返回 Hero 对象
    # FastAPI 用 HeroPublic 过滤和序列化
    # 编辑器不会报警（因为没有返回类型注解的约束）
```

### 5.2 读取列表——返回 HeroPublic 列表

```python
@app.get("/heroes/", response_model=list[HeroPublic])
def read_heroes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes    # 返回 Hero 列表，response_model 过滤成 HeroPublic 列表
```

`response_model=list[HeroPublic]` → 列表里的每个元素都用 HeroPublic 序列化

### 5.3 读取单条——返回 HeroPublic

```python
@app.get("/heroes/{hero_id}", response_model=HeroPublic)
def read_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero
```

和之前一样，`response_model=HeroPublic` 过滤掉 `secret_name`

### 5.4 更新——接收 HeroUpdate，返回 HeroPublic

```python
@app.patch("/heroes/{hero_id}", response_model=HeroPublic)
def update_hero(hero_id: int, hero: HeroUpdate, session: SessionDep):
    hero_db = session.get(Hero, hero_id)
    if not hero_db:
        raise HTTPException(status_code=404, detail="Hero not found")
    hero_data = hero.model_dump(exclude_unset=True)
    hero_db.sqlmodel_update(hero_data)
    session.add(hero_db)
    session.commit()
    session.refresh(hero_db)
    return hero_db
```

逐步拆解：

```
第一步：按 id 从数据库查出旧数据
  hero_db = session.get(Hero, hero_id)
  → Hero(id=1, name="Spider-Boy", age=18, secret_name="Pedro Parqueador")

第二步：提取客户端真正传了的字段（你在第二十二节学过这个！）
  hero_data = hero.model_dump(exclude_unset=True)
  → 如果客户端发了 {"name": "Spider-Man"}
  → hero_data = {"name": "Spider-Man"}（只有 name）

第三步：用新数据更新旧对象
  hero_db.sqlmodel_update(hero_data)
  → hero_db.name = "Spider-Man"（更新了）
  → hero_db.age = 18（没变）
  → hero_db.secret_name = "Pedro Parqueador"（没变）

  sqlmodel_update 是 SQLModel 提供的方法，
  和你之前学的 model_copy(update=...) 效果类似，
  但它直接修改原对象（不创建新对象）。

第四步：add → commit → refresh
  → 变更写入数据库

第五步：返回更新后的数据（response_model 过滤掉 secret_name）
```

这就是第二十二节学的 PATCH 部分更新的实战版本——从字典模拟数据库升级到了真正的 SQL 数据库

### 5.5 删除——不变

```python
@app.delete("/heroes/{hero_id}")
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(hero)
    session.commit()
    return {"ok": True}
```

删除逻辑和单模型版本一样——没有安全问题需要修复

---

## 6. 对应到你的投标项目

```python
from typing import Annotated
from datetime import datetime
from decimal import Decimal

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlmodel import Field, Session, SQLModel, create_engine, select


# ==================== 模型继承体系 ====================

class ProjectBase(SQLModel):
    """基类：所有模型共享的字段"""
    project_name: str = Field(min_length=1, max_length=200, index=True)
    location: str = Field(index=True)
    control_price: float = Field(gt=0, description="控制价，万元")
    project_type: str = Field(description="市政 / 公路 / 房建", index=True)


class Project(ProjectBase, table=True):
    """表模型：对应数据库里的 project 表"""
    id: int | None = Field(default=None, primary_key=True)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str | None = Field(default=None, description="内部备注，不对外公开")


class ProjectPublic(ProjectBase):
    """公开模型：返回给客户端（没有 notes）"""
    id: int
    created_at: str
    updated_at: str


class ProjectCreate(ProjectBase):
    """创建模型：客户端提交的数据（没有 id、没有时间戳）"""
    notes: str | None = None


class ProjectUpdate(SQLModel):
    """更新模型：所有字段可选"""
    project_name: str | None = None
    location: str | None = None
    control_price: float | None = Field(default=None, gt=0)
    project_type: str | None = None
    notes: str | None = None


# ==================== 数据库连接 ====================

engine = create_engine(
    "sqlite:///bidding.db",
    connect_args={"check_same_thread": False},
)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


# ==================== FastAPI 应用 ====================

app = FastAPI(title="智能投标决策系统")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


# ==================== CRUD 接口 ====================

@app.post(
    "/api/projects/",
    response_model=ProjectPublic,
    status_code=status.HTTP_201_CREATED,
    tags=["项目管理"],
)
def create_project(project: ProjectCreate, session: SessionDep):
    """
    创建项目。

    客户端发送 ProjectCreate（没有 id，不能控制 id）
    → 转成 Project 表模型 → 存入数据库
    → 返回 ProjectPublic（没有 notes 内部备注）
    """
    db_project = Project.model_validate(project)
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return db_project


@app.get(
    "/api/projects/",
    response_model=list[ProjectPublic],
    tags=["项目管理"],
)
def list_projects(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
    project_type: str | None = None,
    location: str | None = None,
):
    """查询项目列表，支持按类型和地区筛选"""
    query = select(Project)
    if project_type:
        query = query.where(Project.project_type == project_type)
    if location:
        query = query.where(Project.location.contains(location))
    query = query.offset(offset).limit(limit)
    projects = session.exec(query).all()
    return projects


@app.get(
    "/api/projects/{project_id}",
    response_model=ProjectPublic,
    tags=["项目管理"],
)
def get_project(project_id: int, session: SessionDep):
    """查看单个项目详情"""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


@app.patch(
    "/api/projects/{project_id}",
    response_model=ProjectPublic,
    tags=["项目管理"],
)
def update_project(project_id: int, project: ProjectUpdate, session: SessionDep):
    """
    部分更新项目。

    用了第二十二节学的 exclude_unset 技巧：
    客户端只传要改的字段，其他字段保持不变。
    """
    db_project = session.get(Project, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="项目不存在")
    project_data = project.model_dump(exclude_unset=True)
    if not project_data:
        raise HTTPException(status_code=400, detail="请至少提供一个要更新的字段")
    db_project.sqlmodel_update(project_data)
    db_project.updated_at = datetime.now().isoformat()
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return db_project


@app.delete("/api/projects/{project_id}", tags=["项目管理"])
def delete_project(project_id: int, session: SessionDep):
    """删除项目"""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    session.delete(project)
    session.commit()
    return {"ok": True, "detail": f"项目 {project_id} 已删除"}
```

---

## 7. Session 的三个关键操作速记

```css
操作                   含义                            等价 SQL
────────────────────────────────────────────────────────────────────
session.add(obj)       把对象加入会话（准备写入）        （暂无 SQL，只是标记）
session.commit()       把所有变更写入数据库              COMMIT
session.refresh(obj)   从数据库重新读取对象              SELECT ... WHERE id = ?

session.delete(obj)    标记对象为删除                    （暂无 SQL，只是标记）
session.commit()       执行删除                         DELETE ... WHERE id = ?

session.exec(query)    执行查询                         SELECT ...
  .all()               返回所有结果（列表）
  .first()             返回第一条结果（或 None）
  .one()               返回唯一一条结果（不是一条就报错）

session.get(Model, id) 按主键查询                       SELECT ... WHERE id = ?
```

---

## 8. 本节核心知识点速查

```css
概念                                  做法
──────────────────────────────────────────────────────────────────────────
安装 SQLModel                         pip install sqlmodel
定义表模型                             class Hero(SQLModel, table=True)
定义数据模型（不建表）                  class HeroBase(SQLModel)（没有 table=True）
主键                                  Field(primary_key=True)
索引                                  Field(index=True)
创建引擎                              engine = create_engine("sqlite:///db.db")
创建表                                SQLModel.metadata.create_all(engine)
数据库会话依赖项                       用 yield 的依赖项，with Session(engine) as session
创建记录                              session.add(obj) → commit() → refresh(obj)
查询列表                              session.exec(select(Hero).offset(...).limit(...)).all()
按主键查询                             session.get(Hero, id)
更新记录                              obj.sqlmodel_update(data) → add → commit → refresh
删除记录                              session.delete(obj) → commit()
接收与返回用不同模型                    参数用 HeroCreate，response_model 用 HeroPublic
HeroCreate → Hero 转换                Hero.model_validate(hero_create)
过滤未传字段                           hero.model_dump(exclude_unset=True)
```

**本节最核心的三句话**：

1. **SQLModel 让一个类同时是数据库表和 Pydantic 模型**。加 `table=True` 就是表模型（对应数据库表），不加就是纯数据模型（用于 API 输入输出）。用继承把共享字段提取到基类，避免重复

2. **用 yield 依赖项管理数据库会话**：`with Session(engine) as session: yield session`。每个请求获得一个独立的 Session，请求结束时 Session 自动关闭。这是 FastAPI 连接数据库的标准范式

3. **多模型继承体系是安全的关键**：`HeroCreate`（没有 id → 客户端不能控制 id）、`HeroPublic`（没有 secret_name → 敏感数据不泄露）、`HeroUpdate`（全可选 → 部分更新）。一个模型搞不定，四个模型各司其职