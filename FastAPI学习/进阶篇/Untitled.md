# 🚀 FastAPI + SQLModel 快速上手实战笔记（2026最新版）

> **适用读者**：有 Python 基础、想快速掌握 FastAPI 后端开发的你。
> **目标**：读完本笔记 + 跟着敲完代码，你就能独立搭建一个**带数据库、带认证、带分页、带异步、可生产部署**的 FastAPI 项目。
> **技术栈**：Python 3.11+ / FastAPI 0.115+ / SQLModel 0.0.22+ / SQLAlchemy 2.0+ / Pydantic v2 / Alembic / Uvicorn

------

## 📖 目录

1. [第 1 章：环境准备与项目初始化](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-1-章环境准备与项目初始化)
2. [第 2 章：Pydantic v2 基础（SQLModel 的灵魂）](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-2-章pydantic-v2-基础sqlmodel-的灵魂)
3. [第 3 章：SQLAlchemy 2.0 核心概念](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-3-章sqlalchemy-20-核心概念)
4. [第 4 章：SQLModel 完整讲解](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-4-章sqlmodel-完整讲解)
5. [第 5 章：数据库连接、会话与依赖注入](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-5-章数据库连接会话与依赖注入)
6. [第 6 章：模型定义（表、关系、继承）](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-6-章模型定义表关系继承)
7. [第 7 章：完整 CRUD 实战](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-7-章完整-crud-实战)
8. [第 8 章：高级特性（分页 / 过滤 / 软删除 / 事务 / 异步）](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-8-章高级特性)
9. [第 9 章：FastAPI 完整集成（路由 / 响应模型 / 认证）](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-9-章fastapi-完整集成)
10. [第 10 章：数据库迁移 Alembic](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-10-章数据库迁移-alembic)
11. [第 11 章：最佳实践与常见坑](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-11-章最佳实践与常见坑)

------

## 第 1 章：环境准备与项目初始化

### 1.1 为什么选择 FastAPI + SQLModel？

| 技术               | 作用       | 优势                                                         |
| ------------------ | ---------- | ------------------------------------------------------------ |
| **FastAPI**        | Web 框架   | 性能媲美 Node/Go、自动生成 OpenAPI 文档、类型安全            |
| **Pydantic v2**    | 数据验证   | Rust 核心、比 v1 快 5-50 倍                                  |
| **SQLAlchemy 2.0** | ORM 底座   | Python 最成熟的 ORM，支持同步/异步                           |
| **SQLModel**       | 上层封装   | FastAPI 作者亲自出品，**一个类同时是 Pydantic 模型和数据库表** |
| **Alembic**        | 数据库迁移 | SQLAlchemy 官方迁移工具                                      |

> **一句话总结**：SQLModel = Pydantic + SQLAlchemy 的融合体，让你少写一半代码。

### 1.2 Python 与工具版本要求

```bash
# 检查 Python 版本（需 3.11+，推荐 3.12）
python --version

# 推荐使用 uv（Rust 编写的超快包管理器，2025年已成事实标准）
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> **注意**：如果你不想用 uv，也可以用传统的 `python -m venv` + `pip`，命令会在下面并列给出。

### 1.3 创建项目

```bash
# 创建项目目录（三平台通用）
mkdir fastapi_demo
cd fastapi_demo

# === 方案 A：使用 uv（推荐）===
uv init                          # 初始化 pyproject.toml
uv add fastapi[standard] sqlmodel alembic
uv add --dev pytest httpx ruff   # 开发依赖

# === 方案 B：使用传统 venv ===
python -m venv .venv
# 激活虚拟环境
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

pip install "fastapi[standard]" sqlmodel alembic
pip install pytest httpx ruff
```

### 1.4 推荐的项目目录结构

```css
fastapi_demo/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # 配置（从环境变量读取）
│   │   └── security.py      # 密码哈希 / JWT
│   ├── db/
│   │   ├── __init__.py
│   │   └── session.py       # 数据库引擎与 Session
│   ├── models/              # SQLModel 表模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── post.py
│   ├── schemas/             # Pydantic 请求/响应模型（可选分离）
│   │   ├── __init__.py
│   │   └── user.py
│   ├── crud/                # 数据库操作层
│   │   ├── __init__.py
│   │   └── user.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py          # FastAPI 依赖
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── users.py
│   │       └── posts.py
│   └── tests/
│       └── test_users.py
├── alembic/                 # 迁移文件
├── alembic.ini
├── .env                     # 环境变量
├── pyproject.toml
└── README.md
```

### 实战小结

✅ 你已经学会了项目初始化和依赖安装。
✅ 记住这个目录结构，后面每一章都会往里面塞代码。

### 下一章预告

**Pydantic v2** 是 SQLModel 的上半身，不懂 Pydantic 就无法理解 SQLModel 的数据验证机制。

------

## 第 2 章：Pydantic v2 基础（SQLModel 的灵魂）

### 2.1 Pydantic 是做什么的？

Pydantic 利用 Python 的**类型注解**做**运行时数据验证**。一句话：你声明字段类型，它自动帮你校验数据合法性。

```python
# 文件：examples/pydantic_basic.py

# BaseModel 是 Pydantic 所有模型的基类，提供验证、序列化等全部能力
from pydantic import BaseModel, Field, EmailStr, field_validator
from datetime import datetime
from typing import Optional  # 用于标注"可以是 None"的字段


class UserCreate(BaseModel):
    """用户创建时使用的数据模型。

    这个类既能：
    1. 校验传入的 dict / JSON 数据是否合法；
    2. 自动转换类型（比如字符串 "123" -> int 123）；
    3. 生成漂亮的错误信息。
    """

    # Field(...) 中的 ... 表示"必填"，与默认值 None 相对
    # min_length/max_length 是字符串长度约束
    username: str = Field(..., min_length=3, max_length=32, description="用户名")

    # EmailStr 是 Pydantic 提供的邮箱类型，会自动校验邮箱格式
    # 注意：需要 pip install "pydantic[email]"
    email: EmailStr

    # Optional[int] 等价于 int | None（Python 3.10+ 语法）
    # = None 表示默认值为 None，即非必填
    age: Optional[int] = Field(default=None, ge=0, le=150)  # ge=大于等于, le=小于等于

    # datetime 类型会自动解析 ISO 8601 字符串，如 "2026-04-24T10:00:00"
    created_at: datetime = Field(default_factory=datetime.now)

    # field_validator 是 Pydantic v2 自定义校验器
    # 作用于 username 字段；mode="after" 表示类型转换后再执行
    @field_validator("username", mode="after")
    @classmethod
    def username_must_not_contain_space(cls, v: str) -> str:
        """禁止用户名中包含空格。"""
        if " " in v:
            # Pydantic 会自动捕获 ValueError 并转换为 422 错误响应
            raise ValueError("用户名不能包含空格")
        return v


# === 使用示例 ===
if __name__ == "__main__":
    # 1) 合法数据：自动通过校验
    user = UserCreate(username="alice", email="alice@example.com", age=25)
    print(user)                       # 打印模型对象
    print(user.model_dump())          # 转换为 dict（Pydantic v2 API）
    print(user.model_dump_json())     # 转换为 JSON 字符串

    # 2) 非法数据：抛出 ValidationError
    try:
        UserCreate(username="a b", email="not-an-email", age=-1)
    except Exception as e:
        print(e)
```

> **注意**：Pydantic v1 中的 `.dict()` / `.json()` 在 v2 中改名为 `.model_dump()` / `.model_dump_json()`，老教程的代码可能跑不通，一定要认准 v2 写法

### 2.2 常用字段类型速查表

|             类型              |                示例                 |             说明             |
| :---------------------------: | :---------------------------------: | :--------------------------: |
| `str`, `int`, `float`, `bool` |             `name: str`             |           基础类型           |
|    `list[T]`, `dict[K,V]`     |          `tags: list[str]`          |     Python 3.9+ 泛型写法     |
|  `Optional[T]` / `T | None`   |     `avatar: str | None = None`     |          可为空字段          |
|  `datetime`, `date`, `time`   |       `created_at: datetime`        |           时间类型           |
|            `UUID`             |             `id: UUID`              | 导入 `from uuid import UUID` |
|     `EmailStr`, `HttpUrl`     |          `email: EmailStr`          |          语义化类型          |
|      `Literal["a", "b"]`      | `status: Literal["draft","public"]` |         枚举式字面量         |
|          `Enum` 子类          |                见下                 |           正式枚举           |

### 2.3 嵌套模型与模型配置

```python
# 文件：examples/pydantic_nested.py

from pydantic import BaseModel, ConfigDict
from datetime import datetime


class Address(BaseModel):
    """嵌套子模型：地址。"""
    city: str
    zip_code: str


class User(BaseModel):
    """用户模型（嵌套一个 Address）。"""

    # ConfigDict 是 v2 的新配置方式，替代了 v1 的 class Config
    model_config = ConfigDict(
        # from_attributes=True 允许从 ORM 对象（有 .username 属性的类）中读取数据
        # 这对 SQLModel / SQLAlchemy 返回的对象极其重要！
        from_attributes=True,

        # str_strip_whitespace=True 自动去除字符串两端空白
        str_strip_whitespace=True,

        # extra="forbid" 禁止未声明的字段（多传字段会报错，推荐开启）
        extra="forbid",
    )

    id: int
    username: str
    address: Address     # 嵌套模型，Pydantic 会递归校验
    created_at: datetime


# 使用：从字典创建（支持深层嵌套）
data = {
    "id": 1,
    "username": "alice",
    "address": {"city": "Shanghai", "zip_code": "200000"},
    "created_at": "2026-04-24T10:00:00",
}
user = User(**data)
print(user.address.city)  # 可以直接点语法访问嵌套字段
```

### 实战小结

✅ Pydantic v2 的核心：**类型即校验**，`model_dump()` 序列化，`field_validator` 自定义规则
✅ `from_attributes=True` 是连接 ORM 对象的关键开关
✅ **v2 语法和 v1 大不相同**，看老文章时先确认版本

### 下一章预告

接下来讲 **SQLAlchemy 2.0 核心概念**，这是 SQLModel 的下半身

------

## 第 3 章：SQLAlchemy 2.0 核心概念

### 3.1 SQLAlchemy 架构：Core vs ORM

SQLAlchemy 有两层：

- **Core**：SQL 表达式语言，手写 SQL 的 Pythonic 替代品
- **ORM**：对象关系映射，Python 类 ↔ 数据库表

SQLModel 站在 **ORM** 层之上。但仍需了解几个关键概念

### 3.2 引擎（Engine）—— 连接池的管家

```python
# 文件：examples/sa_engine.py

# create_engine 是 SQLAlchemy 的"数据库引擎"工厂
# 它封装了数据库连接池、方言、URL 等
from sqlalchemy import create_engine

# 数据库 URL 格式：dialect+driver://user:password@host:port/dbname
# SQLite（文件数据库，适合开发）
DATABASE_URL = "sqlite:///./app.db"

# PostgreSQL（生产推荐）
# DATABASE_URL = "postgresql+psycopg://user:pass@localhost:5432/mydb"

# MySQL
# DATABASE_URL = "mysql+pymysql://user:pass@localhost:3306/mydb"

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    # echo=True 会把所有 SQL 打印到控制台，开发调试必备，生产要关闭
    echo=True,

    # connect_args 传给底层 DB-API 的额外参数
    # SQLite 在多线程（如 FastAPI）下必须加 check_same_thread=False
    connect_args={"check_same_thread": False},

    # pool_pre_ping=True 在每次取连接前先 ping 一下，避免死连接
    # 生产环境强烈建议开启
    pool_pre_ping=True,

    # pool_size: 常驻连接数（默认 5）
    # max_overflow: 高峰时最多额外开多少连接（默认 10）
    # pool_size=10, max_overflow=20,  # PostgreSQL/MySQL 推荐配置
)
```

> **注意**：`engine` 对象应在整个应用中**只创建一次**（单例），不要在每次请求里新建

### 3.3 Session —— 工作单元

Session 是你和数据库对话的"会话"。它的职责：

1. 跟踪你修改过的对象
2. 事务管理（提交 / 回滚）
3. 把 Python 对象翻译成 SQL 并执行

```python
# sessionmaker 是 Session 工厂，通常创建一次
from sqlalchemy.orm import sessionmaker, Session

# 创建 Session 工厂
SessionLocal = sessionmaker(
    bind=engine,           # 绑定到引擎
    autoflush=False,       # 关闭自动 flush（推荐手动控制）
    autocommit=False,      # 关闭自动提交（始终显式 commit）
    expire_on_commit=False # 重要：commit 后对象属性不失效，否则访问属性会重新查询
)

# 使用（最佳实践：with 语句自动关闭）
with SessionLocal() as session:
    # 在这里执行所有操作
    # session.add(obj)
    # session.commit()
    pass  # 退出 with 时自动 close
```

### 3.4 声明式模型（SQLAlchemy 2.0 新语法）

SQLAlchemy 2.0 引入了基于 `Mapped[]` 的新语法，更类型友好：

```python
# 文件：examples/sa_model.py

from datetime import datetime
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# 所有模型的基类，必须继承 DeclarativeBase
class Base(DeclarativeBase):
    pass


class User(Base):
    """SQLAlchemy 2.0 风格模型定义。"""

    # __tablename__ 指定数据库表名，不设置会用类名小写
    __tablename__ = "users"

    # Mapped[类型] 表示"映射到数据库列"的字段
    # mapped_column(...) 等价于老语法的 Column(...)
    id: Mapped[int] = mapped_column(primary_key=True)

    # String(50) 指定 VARCHAR(50)；unique/index 是约束
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)

    # Mapped[str | None] 表示"该列可为 NULL"
    email: Mapped[str | None] = mapped_column(String(255))

    # server_default=func.now() 让数据库自动填默认值（而不是 Python）
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
```

> **为什么是 2.0 新语法？** 类型检查（mypy/Pyright）能识别字段类型，IDE 提示更准确，还能自动推断 `Optional`

### 3.5 基本 CRUD 四连击

```python
# 文件：examples/sa_crud.py

from sqlalchemy import select
from examples.sa_engine import engine, SessionLocal
from examples.sa_model import Base, User

# 1) 建表（开发时方便，生产用 Alembic）
Base.metadata.create_all(engine)

with SessionLocal() as session:
    # ========== CREATE ==========
    # 创建一个 Python 对象（此时还未入库）
    new_user = User(username="alice", email="alice@example.com")
    session.add(new_user)     # 加入 session 追踪
    session.commit()          # 写入数据库（生成 INSERT）
    session.refresh(new_user) # 刷新对象，拿到数据库生成的 id 等字段
    print(new_user.id)

    # ========== READ ==========
    # 2.0 推荐用 select() 替代老的 session.query()
    stmt = select(User).where(User.username == "alice")
    user = session.scalars(stmt).first()   # 取第一条；.all() 取全部

    # 主键直接查（最常用）
    user = session.get(User, 1)            # SELECT ... WHERE id=1

    # ========== UPDATE ==========
    user.email = "new@example.com"   # 直接改对象属性
    session.commit()                 # Session 自动生成 UPDATE

    # ========== DELETE ==========
    session.delete(user)
    session.commit()
```

> **核心心智模型**：Session 是一个"暂存区"。你对对象的修改都先在暂存区里，`commit()` 时一次性翻译成 SQL 发给数据库。

### 实战小结

✅ **Engine** 管连接池（全局单例），**Session** 管一次业务操作（请求级别）
✅ SQLAlchemy 2.0 用 `Mapped[]` + `mapped_column()` 声明字段
✅ 查询用 `select()`，主键查用 `session.get()`

### 下一章预告

前面两章做了铺垫，现在进入正题 —— **SQLModel**，你会发现 Pydantic + SQLAlchemy 可以优雅地合为一体

------

## 第 4 章：SQLModel 完整讲解

### 4.1 SQLModel 的核心思想

SQLModel 让**一个类同时扮演两种角色**：

- 加 `table=True`：就是数据库表（SQLAlchemy 模型）
- 不加 `table=True`：就是 Pydantic 模型（请求 / 响应 Schema）

这种**双重身份**大幅减少重复代码

### 4.2 你的第一个 SQLModel

```python
# 文件：app/models/hero.py

from typing import Optional
from sqlmodel import SQLModel, Field


class Hero(SQLModel, table=True):
    """英雄表。

    继承 SQLModel + table=True 参数 = 这是一个数据库表类。
    同时它也是 Pydantic 模型，可以直接用于 FastAPI 请求响应。
    """

    # Field 用法和 Pydantic 几乎一样，但多了数据库相关参数
    # primary_key=True: 主键
    # default=None + Optional[int]: 允许数据库自动生成 id
    id: Optional[int] = Field(default=None, primary_key=True)

    # index=True: 为该字段建立索引，加快查询
    name: str = Field(index=True, max_length=50)

    # 无额外参数的字段：普通 VARCHAR 列
    secret_name: str

    # 可为空字段：Optional + default=None
    age: Optional[int] = Field(default=None, index=True)
```

### 4.3 Field 参数完整速查

|           参数            |        作用         |                     示例                      |
| :-----------------------: | :-----------------: | :-------------------------------------------: |
|       `default=...`       |       默认值        |                  `default=0`                  |
|   `default_factory=...`   |   默认值工厂函数    |        `default_factory=datetime.now`         |
|    `primary_key=True`     |        主键         |      `id: int = Field(primary_key=True)`      |
|       `index=True`        |       建索引        |              用于经常查询的字段               |
|       `unique=True`       |      唯一约束       |                `email` 等字段                 |
| `foreign_key="表名.字段"` |        外键         | `user_id: int = Field(foreign_key="user.id")` |
|   `nullable=True/False`   |      是否可空       |           通常由 Optional 自动推断            |
|      `max_length=N`       |    VARCHAR 长度     |      `name: str = Field(max_length=50)`       |
|      `sa_column=...`      | 直接传 SA 的 Column |                  高级定制用                   |
|  `ge`, `le`, `gt`, `lt`   |  Pydantic 数值约束  |       `age: int = Field(ge=0, le=150)`        |
|    `description="..."`    |      文档描述       |              会显示在 Swagger 中              |

> **注意**：`Optional[int]` + `default=None` 组合起来才是"可空字段"。只写 `Optional[int]` 不给默认值，Pydantic 会认为是**必填但可为 None**

### 4.4 分离"表模型"和"Schema"（重要最佳实践）

真实项目中，**直接用表模型做请求/响应会出事**（比如把 `password_hash` 返回给前端）。推荐拆分：

```python
# 文件：app/models/user.py

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


# ============ 1. 共享基类（不是表）============
class UserBase(SQLModel):
    """共享字段。继承它的子类共享这些字段定义。"""
    username: str = Field(index=True, max_length=50, unique=True)
    email: str = Field(index=True, max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=100)


# ============ 2. 真正的表 ============
class User(UserBase, table=True):
    """数据库表。"""
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str   # 哈希密码，绝不返回给前端
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)


# ============ 3. 创建时接收的数据 ============
class UserCreate(UserBase):
    """POST /users 请求体。比 UserBase 多一个明文 password。"""
    password: str = Field(min_length=8, max_length=64)


# ============ 4. 更新时接收的数据（所有字段可选）============
class UserUpdate(SQLModel):
    """PATCH /users/{id} 请求体。"""
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=8)


# ============ 5. 返回给前端的数据 ============
class UserRead(UserBase):
    """GET /users/{id} 响应体。注意：不包含 hashed_password！"""
    id: int
    is_active: bool
    created_at: datetime
```

> **记住这个五件套**：`Base / Table / Create / Update / Read`，几乎每个资源都照这个套路写

### 4.5 从 Python 类到数据库表

```python
# 文件：app/db/session.py

from sqlmodel import SQLModel, create_engine

# SQLModel 的 create_engine 是对 SQLAlchemy 的薄封装，参数基本一致
engine = create_engine(
    "sqlite:///./app.db",
    echo=True,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    """开发时用这个快速建表，生产用 Alembic。"""
    # import 所有模型，让 SQLModel.metadata 知道它们的存在
    from app.models.user import User  # noqa: F401

    # create_all 只会建**不存在**的表，不会改已有表结构
    SQLModel.metadata.create_all(engine)
```

### 实战小结

✅ SQLModel = Pydantic + SQLAlchemy，`table=True` 是关键开关
✅ 推荐**五件套**拆分：`Base / Table / Create / Update / Read`
✅ 永远不要把带敏感字段的表模型直接当响应模型用

### 下一章预告

下一章讲**会话管理 + 依赖注入**，这是把 SQLModel 嫁接到 FastAPI 的桥梁

------

## 第 5 章：数据库连接、会话与依赖注入

### 5.1 FastAPI 的依赖注入（Dependency Injection）

FastAPI 用 `Depends()` 实现依赖注入。把"获取 Session"写成一个函数，每个路由函数通过 `Depends` 拿到一个全新的 Session

```python
# 文件：app/db/session.py（完整版）

from typing import Generator
from sqlmodel import Session, SQLModel, create_engine

# ===== 引擎（应用级单例）=====
engine = create_engine(
    "sqlite:///./app.db",
    echo=False,                                  # 生产关闭
    connect_args={"check_same_thread": False},   # SQLite 专用
    pool_pre_ping=True,                          # 防死连接
)


def init_db() -> None:
    """应用启动时建表。"""
    from app.models import user, post  # noqa: F401 让 metadata 感知模型
    SQLModel.metadata.create_all(engine)


# ===== 依赖：获取 Session =====
def get_session() -> Generator[Session, None, None]:
    """FastAPI 依赖函数，每个请求拿到一个独立的 Session。

    使用 yield 而不是 return：
    - yield 之前的代码：请求进来时执行（打开 Session）
    - yield 之后的代码：请求结束时执行（关闭 Session）
    这是 FastAPI "依赖 + 清理" 的标准模式。
    """
    # Session(engine) 会从连接池借一个连接
    with Session(engine) as session:
        yield session
    # with 退出时自动 session.close()，连接归还连接池
```

### 5.2 在路由中使用

```python
# 文件：app/api/v1/users.py（片段）

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.user import User, UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}", response_model=UserRead)
def read_user(
    user_id: int,
    # Depends(get_session) 让 FastAPI 每次请求调用一次 get_session()
    # 拿到 Session 对象注入到 session 参数
    session: Session = Depends(get_session),
):
    user = session.get(User, user_id)
    return user
```

### 5.3 类型化依赖（Python 3.11+ 优雅写法）

每个路由都写 `Session = Depends(get_session)` 很啰嗦，用 `Annotated` 抽取成类型别名：

```python
# 文件：app/api/deps.py

from typing import Annotated
from fastapi import Depends
from sqlmodel import Session

from app.db.session import get_session

# 抽取一个类型别名，以后直接用 SessionDep 即可
SessionDep = Annotated[Session, Depends(get_session)]
```

使用：

```python
from app.api.deps import SessionDep

@router.get("/{user_id}", response_model=UserRead)
def read_user(user_id: int, session: SessionDep):  # 清爽很多
    return session.get(User, user_id)
```

> **注意**：`Annotated[X, Depends(...)]` 是 FastAPI 官方 2024 年起力推的写法，可复用、可单元测试时覆盖

### 5.4 应用生命周期事件（启动建表）

FastAPI 推荐用 **lifespan** 而不是老的 `on_event`：

```python
# 文件：app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期上下文。

    yield 之前：应用启动时执行（建表、加载模型等）
    yield 之后：应用关闭时执行（清理资源）
    """
    # 启动逻辑
    init_db()
    print("✅ 数据库已初始化")
    yield
    # 关闭逻辑（可选）
    print("👋 应用关闭")


app = FastAPI(
    title="FastAPI Demo",
    version="1.0.0",
    lifespan=lifespan,  # 挂上生命周期
)
```

### 实战小结

✅ **Engine 全局一个**，**Session 每个请求一个**
✅ `get_session` 用 `yield` 实现自动清理
✅ 用 `Annotated` + `SessionDep` 写出优雅路由
✅ 应用启动用 `lifespan` 上下文管理器

### 下一章预告

下一章讲**模型关系**（一对多、多对多），这是真实项目最高频的场景。

------

## 第 6 章：模型定义（表、关系、继承）

### 6.1 一对多关系：User 与 Post

一个用户可以发多个帖子，一个帖子只属于一个用户。

```python
# 文件：app/models/user.py

from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship

# TYPE_CHECKING 下导入只在类型检查时生效，避免循环 import
if TYPE_CHECKING:
    from app.models.post import Post


class UserBase(SQLModel):
    username: str = Field(index=True, unique=True, max_length=50)
    email: str = Field(index=True, max_length=255)


class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)

    # ========= 关系字段 =========
    # Relationship 不是数据库列，它是一个"便利属性"
    # back_populates 指向对端模型中对应的字段名，必须双向一致
    # 字符串 "Post" 是延迟解析，避免循环 import 问题
    posts: List["Post"] = Relationship(
        back_populates="author",

        # cascade_delete=True：删除 User 时级联删除它的 Post
        # 这是 SQLModel 0.0.14+ 新增的便利参数
        cascade_delete=True,
    )
# 文件：app/models/post.py

from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship

if TYPE_CHECKING:
    from app.models.user import User


class PostBase(SQLModel):
    title: str = Field(max_length=200)
    content: str


class Post(PostBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

    # 外键：指向 user 表的 id 列
    # ondelete="CASCADE" 让数据库层也级联（和上面 cascade_delete 双保险）
    author_id: Optional[int] = Field(
        default=None,
        foreign_key="user.id",
        ondelete="CASCADE",
        index=True,
    )

    # 反向关系：通过 author_id 关联到 User
    author: Optional["User"] = Relationship(back_populates="posts")


# Read Schema（返回时把关系也带上）
class PostRead(PostBase):
    id: int
    author_id: int
    created_at: datetime
```

### 6.2 关系的使用

```python
# 创建时：不用手动设 author_id，赋值关系对象即可
alice = User(username="alice", email="a@x.com", hashed_password="...")
post1 = Post(title="Hello", content="World", author=alice)

session.add(alice)   # 只加父对象，子对象会级联加
session.commit()

# 读取时：像属性一样访问
post = session.get(Post, 1)
print(post.author.username)       # 延迟加载，会触发一次 SELECT users
print(len(post.author.posts))     # 再触发一次 SELECT posts
```

> **N+1 查询问题**：上面这种"循环里访问关系属性"会导致大量 SQL。解决方案见 [第 8 章](https://leopard-x.memofun.net/c/5aac809f-7b48-4916-bf6d-78edae283909#第-8-章高级特性) 的 `selectinload`。

### 6.3 多对多关系：标签系统

一个帖子有多个标签，一个标签被多个帖子使用。需要**中间表**。

```python
# 文件：app/models/tag.py

from typing import List, Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship

if TYPE_CHECKING:
    from app.models.post import Post


class PostTagLink(SQLModel, table=True):
    """中间表。多对多必备。"""
    # 两个外键组成联合主键
    post_id: Optional[int] = Field(
        default=None, foreign_key="post.id", primary_key=True
    )
    tag_id: Optional[int] = Field(
        default=None, foreign_key="tag.id", primary_key=True
    )


class Tag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, max_length=30, index=True)

    # 通过中间表关联到 Post
    posts: List["Post"] = Relationship(
        back_populates="tags",
        # link_model 指定中间表（SQLModel 会自动生成 JOIN）
        link_model=PostTagLink,
    )
```

然后在 `Post` 里加：

```python
# app/models/post.py 追加

from app.models.tag import PostTagLink, Tag  # 注意循环 import 风险

class Post(PostBase, table=True):
    # ... 原有字段 ...

    tags: List["Tag"] = Relationship(
        back_populates="posts",
        link_model=PostTagLink,
    )
```

### 6.4 继承策略：别过度设计

SQLModel 支持 SQLAlchemy 的 **单表继承 / 多表继承**，但 99% 的项目用不上。真实项目里**组合优于继承**：用 `UserBase` 共享字段即可。

### 实战小结

✅ **一对多**：一端用 `Relationship(back_populates=...)`，多端加 `foreign_key`。
✅ **多对多**：必须写中间表 + `link_model` 参数。
✅ `TYPE_CHECKING` + 字符串引用避免循环 import。
✅ `cascade_delete=True` + `ondelete="CASCADE"` 级联双保险。

### 下一章预告

模型搭好了，下一章终于要动手写完整的 **CRUD 接口**。

------

## 第 7 章：完整 CRUD 实战

### 7.1 CRUD 层（与路由解耦）

把数据库操作封装成函数，路由层只负责 HTTP。这样可以在单元测试、命令行脚本中复用。

```python
# 文件：app/crud/user.py

from typing import Optional, Sequence
from sqlmodel import Session, select

from app.models.user import User, UserCreate, UserUpdate
from app.core.security import hash_password  # 下一章讲


# ============ CREATE ============
def create_user(session: Session, data: UserCreate) -> User:
    """创建用户。

    Args:
        session: 数据库会话
        data: 请求体（UserCreate Schema）
    Returns:
        创建好的 User ORM 对象（带 id）
    """
    # 把 Pydantic 模型转成 dict，排除 password 字段
    # model_dump(exclude={"password"}) 是 Pydantic v2 的用法
    user_dict = data.model_dump(exclude={"password"})
    user_dict["hashed_password"] = hash_password(data.password)

    # 用 dict 解包构造 User（SQLModel 支持 **kwargs）
    user = User(**user_dict)

    session.add(user)
    session.commit()
    session.refresh(user)  # 取回数据库生成的 id / created_at
    return user


# ============ READ ============
def get_user(session: Session, user_id: int) -> Optional[User]:
    """按 ID 获取用户。没找到返回 None。"""
    return session.get(User, user_id)


def get_user_by_username(session: Session, username: str) -> Optional[User]:
    """按用户名获取用户。"""
    stmt = select(User).where(User.username == username)
    return session.exec(stmt).first()  # .exec() 是 SQLModel 封装，推荐用它


def list_users(
    session: Session,
    skip: int = 0,
    limit: int = 20,
) -> Sequence[User]:
    """分页获取用户列表。"""
    stmt = select(User).offset(skip).limit(limit).order_by(User.id)
    return session.exec(stmt).all()


# ============ UPDATE ============
def update_user(
    session: Session, user: User, data: UserUpdate
) -> User:
    """部分更新用户。

    data 里只传了的字段会更新，未传的保持原样。
    """
    # exclude_unset=True：只拿用户**显式传入**的字段（关键！）
    # 不加这个参数，默认值 None 会把原有数据清空
    update_data = data.model_dump(exclude_unset=True)

    # 如果传了密码，单独哈希
    if "password" in update_data:
        update_data["hashed_password"] = hash_password(update_data.pop("password"))

    # sqlmodel_update 是 SQLModel 的便利方法
    # 等价于：for k, v in update_data.items(): setattr(user, k, v)
    user.sqlmodel_update(update_data)

    session.add(user)
    session.commit()
    session.refresh(user)
    return user


# ============ DELETE ============
def delete_user(session: Session, user: User) -> None:
    """硬删除。"""
    session.delete(user)
    session.commit()
```

### 7.2 路由层（FastAPI）

```python
# 文件：app/api/v1/users.py

from fastapi import APIRouter, HTTPException, status
from typing import List

from app.api.deps import SessionDep
from app.crud import user as crud_user
from app.models.user import UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "/",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="创建用户",
)
def create_user(data: UserCreate, session: SessionDep):
    """注册新用户。用户名必须唯一。"""
    # 业务校验：用户名是否已存在
    if crud_user.get_user_by_username(session, data.username):
        # 抛出 HTTPException，FastAPI 自动转为 400 响应
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已被占用",
        )
    return crud_user.create_user(session, data)


@router.get("/", response_model=List[UserRead])
def list_users(session: SessionDep, skip: int = 0, limit: int = 20):
    """分页获取用户列表。"""
    # limit 加个上限，防止被刷
    limit = min(limit, 100)
    return crud_user.list_users(session, skip=skip, limit=limit)


@router.get("/{user_id}", response_model=UserRead)
def read_user(user_id: int, session: SessionDep):
    """按 ID 获取用户详情。"""
    user = crud_user.get_user(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@router.patch("/{user_id}", response_model=UserRead)
def update_user(user_id: int, data: UserUpdate, session: SessionDep):
    """部分更新用户信息（PATCH 语义）。"""
    user = crud_user.get_user(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return crud_user.update_user(session, user, data)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, session: SessionDep):
    """删除用户。"""
    user = crud_user.get_user(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    crud_user.delete_user(session, user)
    # 204 响应不返回 body
```

### 7.3 把路由挂到主应用上

```python
# 文件：app/main.py（完整版）

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db.session import init_db
from app.api.v1 import users, posts  # 下一节补 posts


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="FastAPI Demo", lifespan=lifespan)

# 挂载路由
app.include_router(users.router, prefix="/api/v1")
app.include_router(posts.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}
```

### 7.4 启动与测试

```bash
# 开发启动（热重载）
fastapi dev app/main.py

# 或用传统方式
uvicorn app.main:app --reload
```

浏览器打开 `http://127.0.0.1:8000/docs`，你会看到自动生成的 Swagger 文档。直接在页面上点 "Try it out" 就能测试。

### 实战小结

✅ **三层架构**：路由层（HTTP）→ CRUD 层（数据库）→ 模型层。
✅ 更新时**务必用 `exclude_unset=True`**，否则 PATCH 变 PUT。
✅ 404 / 400 等错误通过 `HTTPException` 抛出，FastAPI 自动转换。

### 下一章预告

下一章讲**生产级高级特性**：分页、过滤、软删除、事务、异步。

------

## 第 8 章：高级特性

### 8.1 通用分页

固定 `skip + limit` 太原始。真实项目推荐**统一分页响应结构**。

```python
# 文件：app/schemas/page.py

from typing import Generic, List, TypeVar
from pydantic import BaseModel

# 泛型 TypeVar，让 Page 可以承载任何类型的 item
T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """通用分页响应。

    使用示例：Page[UserRead] 表示一页 UserRead。
    """
    items: List[T]
    total: int           # 总记录数
    page: int            # 当前页码（从 1 开始）
    size: int            # 每页数量
    pages: int           # 总页数
```

分页查询实现：

```python
# 文件：app/crud/user.py 追加

from math import ceil
from sqlmodel import func  # 用于 COUNT(*)

def paginate_users(
    session: Session, page: int = 1, size: int = 20
) -> dict:
    # 计算偏移量
    offset = (page - 1) * size

    # 1) 查当前页数据
    stmt = select(User).offset(offset).limit(size).order_by(User.id.desc())
    items = session.exec(stmt).all()

    # 2) 查总数（注意：要去掉 order_by / limit 的单独 count）
    total = session.exec(select(func.count()).select_from(User)).one()

    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": ceil(total / size) if size else 0,
    }
```

路由层：

```python
from app.schemas.page import Page

@router.get("/", response_model=Page[UserRead])
def paginate(session: SessionDep, page: int = 1, size: int = 20):
    size = min(size, 100)
    return crud_user.paginate_users(session, page, size)
```

### 8.2 动态过滤

```python
# 文件：app/crud/user.py 追加

def search_users(
    session: Session,
    username: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> Sequence[User]:
    """按条件查询用户。只对非 None 的参数添加 WHERE。"""
    stmt = select(User)

    if username is not None:
        # ilike 是不区分大小写的 LIKE（PostgreSQL/SQLite 支持）
        stmt = stmt.where(User.username.ilike(f"%{username}%"))

    if is_active is not None:
        stmt = stmt.where(User.is_active == is_active)

    return session.exec(stmt).all()
```

### 8.3 软删除（逻辑删除）

给表加 `deleted_at` 字段，"删除"只是打个时间戳。

```python
# 文件：app/models/user.py（增强）

class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # ... 其他字段 ...
    deleted_at: Optional[datetime] = Field(default=None, index=True)


# CRUD 里
def soft_delete_user(session: Session, user: User) -> User:
    user.deleted_at = datetime.now()
    session.add(user)
    session.commit()
    return user


# 所有查询自动过滤已删除
def get_active_user(session: Session, user_id: int) -> Optional[User]:
    stmt = select(User).where(User.id == user_id, User.deleted_at.is_(None))
    return session.exec(stmt).first()
```

> **注意**：软删除会让每个查询都得加 `WHERE deleted_at IS NULL`，代码容易遗漏。SQLAlchemy 的**事件钩子**或**查询默认过滤器**可以自动化，但复杂度上升，团队要统一约定。

### 8.4 事务

单个 `commit()` 就是一个事务。需要**多个操作要么全成要么全败**时：

```python
def transfer_post(session: Session, post_id: int, new_owner_id: int):
    """把帖子转让给另一个用户。"""
    try:
        post = session.get(Post, post_id)
        new_owner = session.get(User, new_owner_id)
        if not post or not new_owner:
            raise ValueError("目标不存在")

        post.author_id = new_owner.id
        # 还有其他联动操作...

        session.commit()   # 一次提交所有变更
    except Exception:
        session.rollback()  # 任何异常都回滚
        raise
```

用 `session.begin()` 上下文更清爽：

```python
with session.begin():       # 进入事务
    post.author_id = new_owner.id
    # 任何异常自动 rollback，没异常自动 commit
```

### 8.5 避免 N+1：eager loading

```python
from sqlalchemy.orm import selectinload

# 错误示范：循环访问 .author 会触发 N 次额外 SQL
posts = session.exec(select(Post)).all()
for p in posts:
    print(p.author.username)   # N+1 灾难

# 正确做法：selectinload 一次预加载所有关联
stmt = select(Post).options(selectinload(Post.author))
posts = session.exec(stmt).all()
for p in posts:
    print(p.author.username)   # 内存里直接拿，无额外 SQL
```

**三种 eager loading 策略**：

| 策略             | 场景                     | SQL 特点                     |
| ---------------- | ------------------------ | ---------------------------- |
| `selectinload`   | **一对多 / 多对多** 首选 | 2 条 SQL：主表 + `IN` 查子表 |
| `joinedload`     | 一对一 / 小型多对一      | 1 条 SQL + LEFT JOIN         |
| `contains_eager` | 已有自定义 JOIN          | 复用现成 JOIN                |

### 8.6 异步 SQLModel（高并发必备）

同步版在低并发够用，但高并发下，**异步**让一个 worker 同时处理更多请求。

```python
# 安装异步驱动
# pip install aiosqlite                     # SQLite 异步
# pip install "asyncpg"                     # PostgreSQL 异步
# 文件：app/db/session_async.py

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

# 注意 URL：sqlite -> sqlite+aiosqlite；postgresql -> postgresql+asyncpg
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./app.db"

async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)

# 创建异步 Session 工厂
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db_async() -> None:
    """异步建表。"""
    async with async_engine.begin() as conn:
        # run_sync 让同步的 metadata.create_all 在异步事务中跑
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """异步版依赖。"""
    async with AsyncSessionLocal() as session:
        yield session
```

异步 CRUD：

```python
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User

async def get_user_async(session: AsyncSession, user_id: int):
    # 注意：异步模式下用 session.get(...) 需要 await
    return await session.get(User, user_id)


async def list_users_async(session: AsyncSession):
    stmt = select(User)
    # 异步用 session.exec(stmt)（SQLModel）或 session.execute(stmt)（原生 SA）
    result = await session.exec(stmt)
    return result.all()
```

异步路由：

```python
from typing import Annotated
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

AsyncSessionDep = Annotated[AsyncSession, Depends(get_async_session)]

@router.get("/async/{user_id}")
async def get_user(user_id: int, session: AsyncSessionDep):
    return await get_user_async(session, user_id)
```

> **⚠️ 关键陷阱**：**一条请求链里要么全同步、要么全异步，不能混用**。`def` 路由里不要调用 `await`，`async def` 路由里不要用同步 Session（会阻塞事件循环）。

### 实战小结

✅ 分页用统一的 `Page[T]` 响应。
✅ **`selectinload` 解决 N+1**。
✅ 高并发上 **async + asyncpg/aiosqlite**，但注意不要混用同步异步。
✅ 软删除、事务、动态过滤都是生产项目必备。

### 下一章预告

下一章补齐最后的拼图：**FastAPI 认证、中间件、响应模型高级用法**。

------

## 第 9 章：FastAPI 完整集成

### 9.1 配置管理：Pydantic Settings

```bash
pip install pydantic-settings
# 文件：app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 .env 文件和环境变量自动加载配置。"""

    # model_config 告诉 Pydantic 去哪找配置源
    model_config = SettingsConfigDict(
        env_file=".env",               # 从 .env 读取
        env_file_encoding="utf-8",
        case_sensitive=False,          # 环境变量不区分大小写
        extra="ignore",                # 忽略 .env 里多余的键
    )

    PROJECT_NAME: str = "FastAPI Demo"
    DATABASE_URL: str = "sqlite:///./app.db"
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 天


# 实例化一次，全局共享
settings = Settings()
```

`.env` 示例：

```env
PROJECT_NAME=My Awesome API
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/mydb
SECRET_KEY=super-long-random-string-you-must-change
```

### 9.2 密码哈希

```bash
pip install "passlib[bcrypt]"
# 文件：app/core/security.py

from passlib.context import CryptContext

# CryptContext 封装多种哈希算法，bcrypt 是当前推荐
# deprecated="auto" 让旧算法自动标记为过期（后续可迁移）
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """把明文密码哈希成不可逆字符串。"""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码与哈希是否匹配。"""
    return pwd_context.verify(plain, hashed)
```

### 9.3 JWT 认证

```bash
pip install "python-jose[cryptography]"
# 文件：app/core/security.py 追加

from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt

from app.core.config import settings

ALGORITHM = "HS256"


def create_access_token(
    subject: str | int,
    expires_minutes: int | None = None,
) -> str:
    """生成 JWT。"""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    # payload：sub（主题，通常是 user_id）、exp（过期时间）
    payload = {"sub": str(subject), "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """解码 JWT，成功返回 sub，失败返回 None。"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
```

### 9.4 登录接口 + 当前用户依赖

```python
# 文件：app/api/deps.py 增强

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated
from sqlmodel import Session

from app.core.security import decode_access_token
from app.crud import user as crud_user
from app.db.session import get_session
from app.models.user import User

SessionDep = Annotated[Session, Depends(get_session)]

# tokenUrl 指向登录接口，Swagger 会用它做 "Authorize" 按钮
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login")


def get_current_user(
    session: SessionDep,
    token: Annotated[str, Depends(oauth2_scheme)],
) -> User:
    """从 Authorization: Bearer <token> 中解析当前用户。"""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="认证失败",
        headers={"WWW-Authenticate": "Bearer"},
    )
    user_id = decode_access_token(token)
    if user_id is None:
        raise credentials_exc
    user = crud_user.get_user(session, int(user_id))
    if not user or not user.is_active:
        raise credentials_exc
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
```

登录接口：

```python
# 文件：app/api/v1/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Annotated

from app.api.deps import SessionDep
from app.core.security import create_access_token, verify_password
from app.crud import user as crud_user

router = APIRouter(prefix="", tags=["auth"])


@router.post("/login")
def login(
    # OAuth2PasswordRequestForm 要求请求体是 x-www-form-urlencoded
    # 字段固定为 username / password
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
):
    user = crud_user.get_user_by_username(session, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    token = create_access_token(subject=user.id)
    return {"access_token": token, "token_type": "bearer"}
```

使用 `CurrentUser` 保护路由：

```python
from app.api.deps import CurrentUser

@router.get("/me", response_model=UserRead)
def read_me(current_user: CurrentUser):
    """获取当前登录用户（需要 Authorization 头）。"""
    return current_user
```

### 9.5 CORS 跨域

前端和后端不同域名时必须配 CORS：

```python
# app/main.py

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # 前端地址（精确列表更安全）
        "http://localhost:5173",
    ],
    allow_credentials=True,        # 允许携带 Cookie
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 9.6 全局异常处理

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
```

### 实战小结

✅ 配置用 **pydantic-settings**，从 `.env` 读取。
✅ 密码用 **bcrypt** 哈希，**绝不明文存库**。
✅ 认证用 **JWT + OAuth2PasswordBearer**。
✅ `CurrentUser` 类型别名让权限注入只要写一个参数。

### 下一章预告

开发阶段用 `create_all()` 建表没问题，真实项目改表结构要靠 **Alembic 迁移**。

------

## 第 10 章：数据库迁移 Alembic

### 10.1 为什么需要 Alembic？

- `SQLModel.metadata.create_all(engine)` **只建新表**，不会修改已有表。
- 改个字段类型、加个索引、加个外键，都需要对应的 SQL 脚本。
- Alembic 自动比对模型和数据库，生成迁移脚本，并支持升级 / 回滚。

### 10.2 初始化

```bash
# 在项目根目录执行
alembic init alembic
```

会生成：

```
alembic/
├── versions/          # 每次迁移的脚本
├── env.py             # 关键配置文件
├── script.py.mako
alembic.ini            # CLI 配置
```

### 10.3 配置 env.py（关键）

```python
# 文件：alembic/env.py （只展示关键改动点）

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# === 关键 1：导入你的配置和所有模型 ===
from app.core.config import settings
from sqlmodel import SQLModel
# 必须 import 所有 table=True 的模型，让 metadata 感知到它们
from app.models.user import User       # noqa
from app.models.post import Post       # noqa

config = context.config

# === 关键 2：用环境变量里的 DATABASE_URL 覆盖 alembic.ini ===
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# === 关键 3：指向 SQLModel 的 metadata ===
target_metadata = SQLModel.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # render_as_batch=True 对 SQLite 做 ALTER TABLE 必需
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
```

在迁移脚本里使用 SQLModel 类型：

```python
# 文件：alembic/script.py.mako （顶部导入处加）
import sqlmodel    # 让生成的脚本能用 sqlmodel.AutoString 等类型
```

### 10.4 常用命令

```bash
# 1) 根据模型变化自动生成迁移脚本（最常用）
alembic revision --autogenerate -m "create users table"

# 2) 把数据库升级到最新版本
alembic upgrade head

# 3) 回滚一步
alembic downgrade -1

# 4) 查看当前版本
alembic current

# 5) 查看历史
alembic history
```

### 10.5 自动生成的迁移脚本长啥样

```python
# 文件：alembic/versions/xxxx_create_users_table.py

"""create users table

Revision ID: abc123
Revises:
Create Date: 2026-04-24 10:00:00
"""
from alembic import op
import sqlalchemy as sa
import sqlmodel


def upgrade() -> None:
    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sqlmodel.AutoString(length=50), nullable=False),
        sa.Column("email", sqlmodel.AutoString(length=255), nullable=False),
        sa.Column("hashed_password", sqlmodel.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_username", "user", ["username"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_user_username", "user")
    op.drop_table("user")
```

> **注意**：`--autogenerate` 不是万能的，它检测不到 **列改名、表改名、某些约束修改**。生成后**一定要人工 review 脚本**。

### 10.6 去掉 main.py 里的 create_all

生产环境的建表工作交给 Alembic，去掉 `init_db` 的 `create_all` 调用（或只在 dev 下保留）：

```python
# app/db/session.py
def init_db() -> None:
    # 开发期可以保留，生产改用 alembic upgrade head
    if settings.ENVIRONMENT == "dev":
        SQLModel.metadata.create_all(engine)
```

### 实战小结

✅ 模型一变，跑 `alembic revision --autogenerate` + `alembic upgrade head`。
✅ SQLite 必须 `render_as_batch=True` 才能 ALTER。
✅ 自动生成的脚本 **必须 review**，尤其是重命名场景。

### 下一章预告

最后一章是干货合集：**生产部署、性能调优、常见坑**。

------

## 第 11 章：最佳实践与常见坑

### 11.1 目录结构再推荐

真实中大型项目可以按**领域**（domain）拆分模块：

```
app/
├── core/          # 公共设施（config, security, logging）
├── db/            # 引擎、Session、基类
├── domains/
│   ├── users/
│   │   ├── models.py     # 表 + Schema
│   │   ├── crud.py
│   │   ├── router.py
│   │   └── service.py    # 业务逻辑（可选层）
│   └── posts/
│       └── ...
├── api/           # 路由聚合
└── main.py
```

> 小项目用 `models/ schemas/ crud/ api/` 扁平结构就够；团队 5 人以上、资源超过 10 个时再拆 `domains/`。

### 11.2 常见坑 Top 10

1. **SQLite + 多线程要加 `check_same_thread=False`**，否则 `SQLite objects created in a thread can only be used in that same thread` 报错。
2. **`Optional[int]` 不给默认值** → Pydantic 认为"必填但可 None"，想要选填必须 `Optional[int] = None`。
3. **PATCH 接口忘了 `exclude_unset=True`** → 用户没传的字段被当成 None 覆盖原数据。
4. **N+1 查询** → 循环访问 `.posts` 前，加 `options(selectinload(User.posts))`。
5. **异步代码用了同步 Session** → 阻塞事件循环，性能暴跌。
6. **忘了 `session.refresh(obj)`** → 返回的对象没有数据库生成的 `id` / `created_at`。
7. **`hashed_password` 出现在响应里** → 没有拆分 `User` 和 `UserRead` 模型。
8. **循环 import** → 关系字段用字符串 `"Post"` + `TYPE_CHECKING` 导入。
9. **`--autogenerate` 生成的脚本没 review** → 字段重命名被当成"删列+加列"，生产数据丢失。
10. **SECRET_KEY 硬编码在代码里** → 永远从环境变量读，提交前 `git secrets` 扫一次。

### 11.3 性能建议

- **连接池**：生产用 PostgreSQL，`pool_size=10, max_overflow=20` 起步，结合并发量调。
- **索引**：WHERE、ORDER BY、JOIN 涉及的列都应加 `index=True`；但**写多的表少加索引**。
- **批量操作**：插入 1000 条用 `session.add_all(list)`，别用循环 `session.add` + `commit`。
- **分页用 keyset**：超大表 `OFFSET 100000` 极慢，改用 `WHERE id > last_id LIMIT 20`。
- **N+1 根治**：关键 API 必须 `selectinload` 预加载。
- **缓存**：热点数据上 **Redis**，FastAPI 配 `fastapi-cache2`。

### 11.4 测试模板

```python
# 文件：app/tests/test_users.py

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.main import app
from app.db.session import get_session


@pytest.fixture(name="session")
def session_fixture():
    # 内存 SQLite，每个测试用例独立
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    # 用测试 Session 覆盖掉真实依赖
    def get_session_override():
        yield session
    app.dependency_overrides[get_session] = get_session_override
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_create_user(client):
    resp = client.post(
        "/api/v1/users/",
        json={
            "username": "alice",
            "email": "alice@x.com",
            "password": "Secret123!",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "alice"
    assert "hashed_password" not in data  # 关键：敏感字段不能外泄
```

运行：`pytest -v`。

### 11.5 部署简述

- **容器化**：写个 `Dockerfile`，基础镜像用 `python:3.12-slim`。
- **WSGI/ASGI 服务器**：生产用 **Uvicorn + Gunicorn**（`gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4`）。
- **反向代理**：Nginx / Caddy，处理 HTTPS、静态文件、限流。
- **日志**：用 `logging` + JSON 格式输出到 stdout，交给 Docker / K8s 收集。
- **健康检查**：加一个 `/healthz` 路由，返回 DB 连接状态。

### 11.6 学完后你该做什么？

1. **照着本文档从零搭一个"博客后端"**：User + Post + Tag + 评论 + JWT。
2. 把项目**部署到云**（Fly.io、Render、Railway 免费额度很慷慨）。
3. 加**单元测试**，覆盖率冲到 80%。
4. 读一遍 SQLModel、FastAPI、SQLAlchemy 2.0 的**官方文档**（SQLModel 文档特别优秀，一天能读完）。
5. 有精力可以研究 **Celery / Redis 队列 / WebSocket / GraphQL 集成**。

### 最终小结

你现在已经掌握了：

- ✅ Pydantic v2 数据校验
- ✅ SQLAlchemy 2.0 Core / ORM
- ✅ SQLModel 双重身份模型
- ✅ FastAPI 路由 / 依赖注入 / 响应模型
- ✅ 数据库 Session / 事务 / 异步
- ✅ JWT 认证 + 密码哈希
- ✅ Alembic 数据库迁移
- ✅ 分页 / 过滤 / 软删除 / N+1 优化
- ✅ 项目分层架构 + 测试 + 部署

**恭喜你已经是一个合格的 FastAPI 后端工程师了！🎉**

> 最后送一句老话：**不要只收藏不动手**。关掉这份笔记，打开终端，敲一遍 User + Post 的完整 CRUD 吧。