# FastAPI 学习笔记（二十九）：安全性——OAuth2 基础与获取当前用户

安全性听起来很吓人——OAuth2、OpenID Connect、Bearer Token、JWT……一堆术语。但 FastAPI 把这些东西封装得非常优雅：**安全认证本质上就是依赖注入**



你已经掌握了依赖注入的全部知识（子依赖项、yield、装饰器依赖、全局依赖），现在只需要学一个新东西：`OAuth2PasswordBearer`——它就是一个**特殊的依赖项**，自动从请求头里提取 Bearer Token

> **这一节目标**：理解 OAuth2 的 password 流是怎么回事，用 `OAuth2PasswordBearer` + `get_current_user` 依赖项搭建认证骨架

---

## 1. 先搞清楚这些术语——不用深究，知道是什么就行

### 1.1 OAuth2

OAuth2 是一个**规范**（不是一个库，不是一个工具），它定义了"如何处理认证和授权"的多种方式。

你每天都在用它：

```css
"使用 Google 登录"  → OAuth2
"使用 GitHub 登录"  → OAuth2
"使用微信登录"      → OAuth2（的变体）
```

OAuth2 定义了多种"流"（flow），每种流适合不同的场景：

```css
流                    适用场景                                  我们用不用
─────────────────────────────────────────────────────────────────────────
password              前端直接发用户名密码给后端                  ✅ 我们用这个
authorizationCode     "使用 Google/GitHub 登录"那种跳转流程       暂不用
clientCredentials     服务器对服务器（没有用户参与）               暂不用
implicit              已废弃，不推荐                              不用
```

**我们这一节用的是 password 流**——最简单直接的方式：前端发用户名密码，后端返回 Token

### 1.2 OpenID Connect

OpenID Connect 是在 OAuth2 基础上扩展的规范，主要规定了"用户信息怎么返回"等细节。Google 登录用的就是 OpenID Connect。**你现在不需要关心它。** 等以后需要接 Google/GitHub 第三方登录时再看

### 1.3 Bearer Token

```css
Bearer = 持有者

Bearer Token 的意思是：谁持有这个 Token，谁就有权限访问。
（类比：谁拿着门禁卡，谁就能进门。不管你是不是卡的主人。）

它通过 HTTP 请求头传递：
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 1.4 OpenAPI 与安全方案

FastAPI 基于 OpenAPI 规范。OpenAPI 定义了几种安全方案：

```css
安全方案          传递方式               示例
──────────────────────────────────────────────────────
apiKey            查询参数/头部/Cookie    ?api_key=xxx 或 X-API-Key: xxx
http - bearer     Authorization 头部     Authorization: Bearer xxx     ← 我们用这个
http - basic      Authorization 头部     Authorization: Basic base64(user:pass)
oauth2            多种流                 password 流、authorization code 流等
openIdConnect     自动发现               自动从配置 URL 获取
```

**我们用的是 OAuth2 password 流 + Bearer Token**。FastAPI 的 `OAuth2PasswordBearer` 类就是为此设计的

### 1.5 术语总结——一句话版本

```css
OAuth2          → 一套认证授权的规范（不是工具）
password 流     → OAuth2 的一种方式：前端直接发用户名密码，后端返回 Token
Bearer Token    → 通过 Authorization 头传递的令牌
OpenAPI         → API 文档规范，FastAPI 基于它，安全方案会自动显示在文档里
```

记住这些就够了，不需要去读 OAuth2 的完整规范

---

## 2. password 流的完整过程——先看全景

在写代码之前，先理解整个认证流程：

```css
┌─────────────┐                              ┌─────────────┐
│   前端/客户端  │                              │   FastAPI 后端 │
└──────┬──────┘                              └──────┬──────┘
       │                                            │
       │  ① POST /token                             │
       │     Body: username=alice&password=123456    │
       │ ──────────────────────────────────────────► │
       │                                            │  验证用户名密码
       │  ② 返回 Token                               │
       │     {"access_token": "xxx", "token_type": "bearer"}
       │ ◄────────────────────────────────────────── │
       │                                            │
       │  ③ 保存 Token（存在内存/localStorage 里）     │
       │                                            │
       │  ④ 请求受保护的接口                           │
       │     GET /users/me                           │
       │     Authorization: Bearer xxx               │
       │ ──────────────────────────────────────────► │
       │                                            │  从 Authorization 头提取 Token
       │                                            │  用 Token 查找用户
       │  ⑤ 返回用户数据                              │
       │     {"username": "alice", "email": "..."}   │
       │ ◄────────────────────────────────────────── │
```

这一节先实现 ④ 和 ⑤（从请求头提取 Token → 获取当前用户）。下一节再实现 ① 和 ②（登录接口，验证用户名密码，生成 Token）

---

## 3. OAuth2PasswordBearer——一个特殊的依赖项

### 3.1 最简代码

```python
from typing import Annotated
from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

# 创建 OAuth2PasswordBearer 实例
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):
    return {"token": token}
```

就这么几行，你的 API 就有了基本的安全机制

### 3.2 逐行拆解

```python
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
```

这一行做了三件事：

```css
1. 创建一个可调用的依赖项对象（和你写的依赖项函数一样，可以用 Depends 包装）
2. 告诉 OpenAPI 文档："获取 Token 的 URL 是 /token"
   → 这样 Swagger 文档的 Authorize 按钮就知道往哪里发登录请求
3. 不会创建 /token 端点——只是声明它存在，真正的端点你得自己写
```

`tokenUrl="token"` 是**相对 URL**：

```
如果你的 API 在 https://example.com/       → Token URL 是 https://example.com/token
如果你的 API 在 https://example.com/api/v1/ → Token URL 是 https://example.com/api/v1/token
```

```python
@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):
    return {"token": token}
```

`Depends(oauth2_scheme)` 做的事情：

```css
1. 查看请求的 Authorization 头部
2. 检查值是否是 "Bearer <token>" 格式
3. 如果是 → 提取 token 字符串，返回给函数参数
4. 如果不是（没有 Authorization 头，或格式不对）→ 直接返回 401 Unauthorized

你的函数里拿到的 token 一定是一个有效格式的字符串。
不需要自己检查 Authorization 头是否存在。
```

### 3.3 OAuth2PasswordBearer 的本质

```css
OAuth2PasswordBearer 本质上是一个依赖项，等价于你自己写：

async def oauth2_scheme(authorization: str = Header()):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, ...)
    token = authorization.removeprefix("Bearer ")
    return token

只不过 FastAPI 官方实现了这个类，额外做了：
  - 自动集成到 OpenAPI 文档（Swagger 显示锁图标和 Authorize 按钮）
  - 遵循 OAuth2 规范的标准格式
  - 处理各种边界情况
```

所以你在之前的依赖项笔记里学到的所有概念——子依赖项、缓存、装饰器依赖——全部适用于 `oauth2_scheme`。它就是一个普通的依赖项，只是带了 OpenAPI 集成

### 3.4 文档里发生了什么

运行这个代码后，访问 `http://127.0.0.1:8000/docs`：

```css
1. 页面右上角出现 "Authorize" 按钮
2. 每个受保护的路径操作右上角有一个锁图标
3. 点击 Authorize 按钮 → 弹出登录表单（用户名、密码）
4. 点击 Authorize 后，后续的请求会自动带上 Authorization: Bearer <token> 头

这一切都是因为 OAuth2PasswordBearer 告诉了 OpenAPI 安全方案的信息。
```

---

## 4. 获取当前用户——从 Token 到 User 对象

### 4.1 问题：光有 Token 字符串不够

```python
# 每个接口都要自己解析 Token → 重复代码
@app.get("/users/me")
async def read_users_me(token: Annotated[str, Depends(oauth2_scheme)]):
    user = decode_token(token)   # 每个接口都要写这一行
    if not user:                 # 每个接口都要写验证
        raise HTTPException(...)
    return user

@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):
    user = decode_token(token)   # 又写了一遍
    if not user:                 # 又写了一遍
        raise HTTPException(...)
    return get_user_items(user)
```

### 4.2 解决：创建 get_current_user 依赖项

```python
from typing import Annotated
from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# ===== 用户模型 =====
class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


# ===== 模拟的 Token 解码函数 =====
def fake_decode_token(token):
    """实际项目中这里会用 JWT 解码"""
    return User(
        username=token + "fakedecoded",
        email="john@example.com",
        full_name="John Doe",
    )


# ===== 核心依赖项：从 Token 获取当前用户 =====
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    user = fake_decode_token(token)
    return user


# ===== 路径操作：直接拿到 User 对象 =====
@app.get("/users/me")
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user
```

### 4.3 依赖链分析

```css
read_users_me
    │
    │ Depends(get_current_user)
    ▼
get_current_user
    │
    │ Depends(oauth2_scheme)
    ▼
OAuth2PasswordBearer
    │
    │ 从 Authorization 头提取
    ▼
"Bearer eyJhbGci..." → "eyJhbGci..."（去掉 Bearer 前缀的 token 字符串）
```

执行顺序：

```css
① OAuth2PasswordBearer 从请求头提取 Token → "eyJhbGci..."
   （如果没有 Authorization 头 → 直接 401，后面都不执行）

② get_current_user 接收 token 字符串
   → 调用 fake_decode_token 解码
   → 返回 User 对象

③ read_users_me 接收 User 对象
   → 直接使用，不需要知道 Token 是怎么来的
```

### 4.4 这就是依赖注入的威力

```python
# 任何需要当前用户的接口，只需要加一个参数：
@app.get("/users/me")
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user

@app.get("/items/")
async def read_items(current_user: Annotated[User, Depends(get_current_user)]):
    return get_user_items(current_user)

@app.post("/projects/")
async def create_project(
    project: ProjectCreate,
    current_user: Annotated[User, Depends(get_current_user)],
):
    project.created_by = current_user.username
    ...
```

**认证逻辑只写一次**（在 `get_current_user` 里），所有接口复用。修改认证方式（比如从假的 Token 换成真的 JWT）只需要改 `get_current_user` 一个地方

### 4.5 创建类型别名——让代码更简洁

```python
# 定义一次
UserDep = Annotated[User, Depends(get_current_user)]

# 到处用
@app.get("/users/me")
async def read_users_me(current_user: UserDep):
    return current_user

@app.get("/items/")
async def read_items(current_user: UserDep):
    return get_user_items(current_user)
```

这就是之前学的 `Annotated` 类型别名技巧，在安全认证场景下特别有用

---

## 5. 理解 Pydantic 模型在这里的角色

### 5.1 为什么用 Pydantic 模型

```python
class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None
```

好处：

```
1. 类型检查——IDE 能自动补全 current_user.username
2. 数据验证——确保用户对象有正确的字段
3. 文档生成——Swagger 文档自动显示 User 的结构
4. 灵活性——你可以用任何模型，不限于这个结构
```

### 5.2 你可以用任何模型

FastAPI 不强制你用特定的用户模型。根据你的需求自由定义：

```python
# 简单版：只要一个 ID
class User(BaseModel):
    id: int

# 完整版：带角色和权限
class User(BaseModel):
    id: int
    username: str
    email: str
    role: str                    # "admin" / "user" / "viewer"
    permissions: list[str]       # ["read", "write", "delete"]

# 甚至可以直接用字符串
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    return token  # 直接返回 token 字符串作为"用户标识"

# 或者用 SQLModel 数据库模型
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: SessionDep,
) -> User:  # 这里的 User 可以是 SQLModel 的 table=True 模型
    user = session.exec(select(User).where(User.token == token)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Token 无效")
    return user
```

---

## 6. 当前的认证骨架——还缺什么

### 6.1 我们现在有什么

```
✅ OAuth2PasswordBearer：从 Authorization 头提取 Token
✅ get_current_user：把 Token 转换成 User 对象
✅ 依赖注入复用：任何接口都能轻松获取当前用户
✅ OpenAPI 集成：Swagger 文档有 Authorize 按钮和锁图标
```

### 6.2 还缺什么

```
❌ 登录接口（POST /token）：接收用户名密码，返回 Token
❌ 密码哈希：不能明文存密码
❌ 真正的 Token 生成：目前 fake_decode_token 是假的
❌ Token 过期机制：Token 不能永远有效
```

这些会在下一节学习

### 6.3 当前代码的完整依赖树

```css
FastAPI 应用
│
├── OAuth2PasswordBearer(tokenUrl="token")
│   └── 功能：从 Authorization: Bearer xxx 头提取 token 字符串
│   └── 失败：返回 401 Unauthorized
│
├── get_current_user(token)
│   └── 依赖：oauth2_scheme
│   └── 功能：把 token 解码成 User 对象
│   └── 失败：（目前没处理，下一节加）
│
├── GET /users/me
│   └── 依赖：get_current_user → 拿到 User 对象
│
└── GET /items/
    └── 依赖：get_current_user → 拿到 User 对象
```

---

## 7. 对应到你的投标项目

### 7.1 现阶段的代码

```python
# app/security.py（新文件，专门放安全相关的东西）

from fastapi.security import OAuth2PasswordBearer

# tokenUrl 用相对路径，配合 prefix 使用
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
```

```python
# app/dependencies.py（在已有的基础上修改）

from typing import Annotated
from fastapi import Depends, HTTPException
from sqlmodel import Session, select

from .database import get_session
from .security import oauth2_scheme
from .models import User

SessionDep = Annotated[Session, Depends(get_session)]


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: SessionDep,
):
    """从 Token 获取当前用户——目前是模拟实现，下一节替换成 JWT"""
    # TODO: 真正的 JWT 解码
    user = session.exec(select(User).where(User.username == token)).first()
    if not user:
        raise HTTPException(status_code=401, detail="认证失败")
    return user

UserDep = Annotated[User, Depends(get_current_user)]
```

```python
# app/routers/projects.py（加上认证）

from fastapi import APIRouter
from ..dependencies import SessionDep, UserDep

router = APIRouter(prefix="/projects", tags=["项目管理"])


@router.get("/")
async def list_projects(session: SessionDep, current_user: UserDep):
    """需要登录才能查看项目列表"""
    return session.exec(select(Project)).all()


@router.post("/", status_code=201)
async def create_project(
    project: ProjectCreate,
    session: SessionDep,
    current_user: UserDep,       # 用于记录谁创建的
):
    db_project = Project.model_validate(project)
    db_project.created_by = current_user.username
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return db_project
```

### 7.2 依赖链

```
请求到达 POST /api/projects/

① oauth2_scheme
   → 从 Authorization: Bearer xxx 提取 token

② get_session（yield 依赖项，缓存共享）
   → 创建数据库会话

③ get_current_user（子依赖项）
   → 用 token 查数据库找用户
   → 找不到 → 401
   → 找到 → 返回 User 对象

④ create_project 执行
   → 用 current_user.username 记录创建者
   → 用 session 操作数据库

⑤ 请求结束
   → get_session 的 yield 之后执行，关闭数据库连接
```

---

## 8. 本节核心知识点速查

```css
概念                                    做法
──────────────────────────────────────────────────────────────────────────
OAuth2 password 流                      前端发用户名密码 → 后端返回 Token → 后续请求带 Token
OAuth2PasswordBearer                    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
提取 Token                              token: Annotated[str, Depends(oauth2_scheme)]
获取当前用户                            get_current_user 依赖项，子依赖 oauth2_scheme
类型别名                                UserDep = Annotated[User, Depends(get_current_user)]
OpenAPI 集成                            自动出现 Authorize 按钮和锁图标
tokenUrl 是相对路径                      tokenUrl="token" → 相对于 API 根路径
没有 Authorization 头                    自动返回 401 Unauthorized
```

**本节最核心的三句话**：

1. **OAuth2PasswordBearer 就是一个特殊的依赖项**。它从请求的 `Authorization: Bearer xxx` 头里提取 Token 字符串并返回。没有 Authorization 头就自动返回 401。你用 `Depends(oauth2_scheme)` 就像用任何其他依赖项一样

2. **认证的核心是 `get_current_user` 依赖项链**。`oauth2_scheme` 提取 Token → `get_current_user` 把 Token 转换成 User 对象 → 路径操作函数直接拿到 User。这条依赖链写一次，所有接口复用。修改认证方式只改一个地方

3. **安全认证 = 依赖注入 + OpenAPI 集成**。你之前学的所有依赖注入知识（子依赖项、缓存、装饰器依赖、全局依赖、yield）在安全认证场景下全部适用。`OAuth2PasswordBearer` 唯一特别的地方是它额外把安全方案信息写进了 OpenAPI 文档，让 Swagger 自动显示 Authorize 按钮

---

## 下一步

目前的认证是"骨架"——`fake_decode_token` 是假的，也没有登录接口。下一节将实现完整的认证流程：用户名密码验证、密码哈希（bcrypt）、JWT Token 生成与解码、Token 过期机制。到那时，你的投标系统就有了真正可用的安全认证
