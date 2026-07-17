上一篇补充了子依赖项和装饰器依赖项。这一篇补上最后两块：

​	**全局依赖项**——一行代码让所有接口都必须通过某个验证。这个概念很简单，几分钟就懂

​	**yield 依赖项**——依赖注入体系里最精妙的设计。yield 依赖项解决的核心问题是：**资源的创建和清理**。数据库连接要关闭、文件句柄要关闭、临时文件要删除、锁要释放……yield 之前创建资源，yield 把资源交给路径操作函数用，yield 之后清理资源。
**无论请求成功还是异常，清理代码都保证执行**

---

## 1. 全局依赖项——最简单但最常用

### 1.1 一行代码搞定

```python
from typing import Annotated
from fastapi import Depends, FastAPI, Header, HTTPException


async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: Annotated[str, Header()]):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key


# 关键就这一行：dependencies 参数
app = FastAPI(dependencies=[Depends(verify_token), Depends(verify_key)])


@app.get("/items/")
async def read_items():
    # 不需要声明任何依赖参数
    # verify_token 和 verify_key 已经自动执行了
    return [{"item": "Portal Gun"}, {"item": "Plumbus"}]


@app.get("/users/")
async def read_users():
    # 这个接口也自动执行 verify_token 和 verify_key
    return [{"username": "Rick"}, {"username": "Morty"}]
```

### 1.2 全局依赖项的特点

```css
1. 应用于所有路径操作——包括 main.py 里定义的和所有 include_router 引入的
2. 本质是装饰器依赖——返回值不会传给任何路径操作函数
3. 可以有子依赖项——verify_token 可以依赖其他函数
4. 验证失败会拦截请求——和其他依赖项一样，抛异常就阻止请求继续
```

### 1.3 适用场景

```
场景                              示例
──────────────────────────────────────────────────────
API Key 验证                     所有接口都需要有效的 API Key
请求日志记录                      记录每个请求的时间、路径、来源 IP
频率限制                          限制每个客户端的请求频率
多租户识别                        从请求头里提取租户 ID
```

### 1.4 三个层级的完整回顾

把上一篇和这一篇的内容拼在一起，你现在掌握了依赖项的所有层级：

```python
# 层级一：全局——所有接口
app = FastAPI(dependencies=[Depends(verify_api_key)])

# 层级二：Router——该模块所有接口
router = APIRouter(dependencies=[Depends(verify_token)])
# 或
app.include_router(router, dependencies=[Depends(verify_token)])

# 层级三：路径操作——单个接口（装饰器依赖，不需要返回值）
@router.get("/admin", dependencies=[Depends(require_admin)])

# 层级四：参数——单个接口（参数依赖，需要返回值）
@router.get("/me")
async def get_me(user: Annotated[dict, Depends(get_current_user)]):
```

执行顺序：全局 → Router → 路径操作装饰器 → 参数。任何一层失败，后面的都不执行

---

## 2. yield 依赖项——先创建、后清理

### 2.1 为什么需要 yield

先回顾普通依赖项的问题：

```python
# 普通依赖项（用 return）
def get_session():
    session = Session(engine)
    return session
    # 请求结束后，session 怎么关闭？
    # 答案：没人管它。连接泄漏！
```

用 yield 解决：

```python
# yield 依赖项
def get_session():
    session = Session(engine)
    try:
        yield session       # ← 把 session 交给路径操作函数
    finally:
        session.close()     # ← 请求结束后，无论成功还是异常，都关闭 session
```

### 2.2 yield 依赖项的执行时机

```
请求到达
    │
    ▼
yield 之前的代码执行（创建资源）
    session = Session(engine)
    │
    ▼
yield session（把资源交出去）
    │
    ▼
路径操作函数执行（使用资源）
    heroes = session.exec(select(Hero)).all()
    │
    ▼
路径操作函数返回
    │
    ▼
yield 之后的代码执行（清理资源）  ← 默认在响应发送给客户端之后
    session.close()
```

用一句话概括：**yield 之前是"准备"，yield 本身是"交接"，yield 之后是"清理"**

### 2.3 try/finally 是关键

```python
def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()    # finally 保证：无论路径操作函数成功还是抛异常，都会执行
```

为什么用 `try/finally` 而不是直接写？

```python
# ❌ 不安全
def get_session():
    session = Session(engine)
    yield session
    session.close()       # 如果路径操作函数抛了异常，这行不会执行！

# ✅ 安全
def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()   # finally 块无论如何都会执行
```

`finally` 是 Python 的保证——不管 try 块里发生了什么（正常返回、抛异常、甚至 return），finally 块都一定会执行

### 2.4 和 Python 的 with 语句是同一个原理

```python
# Python 的 with 语句（你已经很熟了）
with open("file.txt") as f:
    content = f.read()
# with 结束时自动关闭文件

# yield 依赖项做的事情一模一样
def get_file():
    f = open("file.txt")
    try:
        yield f
    finally:
        f.close()
```

事实上你可以直接在 yield 依赖项里用 with：

```python
def get_session():
    with Session(engine) as session:    # with 负责创建和关闭
        yield session                    # yield 负责把 session 交出去
    # with 块结束时 session 自动关闭
```

这就是第二十七节数据库会话依赖项的写法。`with` 和 `yield` 配合，代码最简洁

---

## 3. yield 依赖项与异常处理——三种模式

### 3.1 模式一：finally——无论如何都清理（最常用）

```python
def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()     # 不管发生什么，都关闭连接
```

这是 90% 的情况。你只想确保资源被清理，不关心具体发生了什么异常

### 3.2 模式二：except + 转换异常——捕获异常并替换成 HTTPException

```python
class OwnerError(Exception):
    pass


def get_username():
    try:
        yield "Rick"
    except OwnerError as e:
        # 捕获特定异常，转换成 HTTP 错误响应
        raise HTTPException(status_code=400, detail=f"Owner error: {e}")


@app.get("/items/{item_id}")
def get_item(item_id: str, username: Annotated[str, Depends(get_username)]):
    if item_id not in data:
        raise HTTPException(status_code=404, detail="Item not found")
    item = data[item_id]
    if item["owner"] != username:
        raise OwnerError(username)    # 这个异常会被依赖项的 except 捕获
    return item
```

执行流程：

```
1. get_username 执行到 yield "Rick"，把 "Rick" 交给 get_item
2. get_item 发现 item 的 owner 不是 username
3. get_item 抛出 OwnerError
4. 这个异常"冒泡"回到 get_username 的 try 块
5. except OwnerError 捕获了它
6. 转换成 HTTPException(400)，客户端收到 400 错误
```

这个模式适合在依赖项层面做**统一的异常转换**——把业务异常转成 HTTP 错误。

### 3.3 模式三：except + 必须重新 raise——千万别吞掉异常

```python
# ❌ 危险！吞掉了异常
def get_username():
    try:
        yield "Rick"
    except InternalError:
        print("出错了，但我没有 raise")
        # 异常被吞掉了！
        # 客户端收到 500 但服务器没有错误日志
        # 你永远不知道出了什么问题
```

```python
# ✅ 正确！捕获后重新抛出
def get_username():
    try:
        yield "Rick"
    except InternalError:
        print("出错了，记录日志后重新抛出")
        raise    # ← 重新抛出原始异常，让 FastAPI 正常处理
```

**规则**：在 yield 依赖项的 except 块里，要么 raise 一个新的 HTTPException（模式二），要么 `raise` 重新抛出原始异常。**绝对不要默默吞掉异常**

### 3.4 三种模式的速查

```
模式                用法                              场景
────────────────────────────────────────────────────────────────
finally             try: yield ... finally: cleanup   只需要确保资源被清理
except + 转换       except XError: raise HTTPException 把业务异常转成 HTTP 错误
except + 重新抛出   except XError: log(); raise       记录日志后让 FastAPI 处理
```

---

## 4. 带 yield 的子依赖项——清理顺序自动管理

### 4.1 多层 yield 依赖项

```python
async def dependency_a():
    dep_a = generate_dep_a()
    try:
        yield dep_a
    finally:
        dep_a.close()


async def dependency_b(dep_a: Annotated[DepA, Depends(dependency_a)]):
    dep_b = generate_dep_b()
    try:
        yield dep_b
    finally:
        dep_b.close(dep_a)    # 注意：清理 b 时还能用 a


async def dependency_c(dep_b: Annotated[DepB, Depends(dependency_b)]):
    dep_c = generate_dep_c()
    try:
        yield dep_c
    finally:
        dep_c.close(dep_b)    # 注意：清理 c 时还能用 b
```

### 4.2 执行顺序（最关键的部分）

```
创建阶段（yield 之前）——从底层到顶层：
  ① dependency_a 执行到 yield → 创建 dep_a
  ② dependency_b 执行到 yield → 创建 dep_b（可以用 dep_a）
  ③ dependency_c 执行到 yield → 创建 dep_c（可以用 dep_b）

路径操作函数执行（使用 dep_c）

清理阶段（yield 之后）——从顶层到底层（反序！）：
  ④ dependency_c 的 finally 执行 → dep_c.close(dep_b)  ← b 还在
  ⑤ dependency_b 的 finally 执行 → dep_b.close(dep_a)  ← a 还在
  ⑥ dependency_a 的 finally 执行 → dep_a.close()
```

**清理顺序和创建顺序相反**——这正是你需要的。清理 C 的时候 B 还没被清理（还能用），清理 B 的时候 A 还没被清理。就像堆积木：先放的最后拿。

这和 Python 的 `with` 嵌套行为完全一致：

```python
with open("a.txt") as a:           # 最先打开
    with open("b.txt") as b:       # 第二个打开
        with open("c.txt") as c:   # 最后打开
            ...                    # 使用 a, b, c
        # c 最先关闭
    # b 第二个关闭
# a 最后关闭
```

### 4.3 实际场景：数据库事务

```python
def get_session():
    with Session(engine) as session:
        yield session
    # session 关闭


def get_transaction(session: Annotated[Session, Depends(get_session)]):
    try:
        yield session    # 把同一个 session 交出去
        session.commit() # 如果路径操作函数没有抛异常，提交事务
    except Exception:
        session.rollback()  # 如果抛了异常，回滚事务
        raise               # 重新抛出异常，让 FastAPI 处理


@app.post("/transfer/")
def transfer_money(
    session: Annotated[Session, Depends(get_transaction)],
):
    # 在这里执行多个数据库操作
    # 如果任何一个失败，get_transaction 的 except 会回滚所有操作
    session.exec(...)  # 扣款
    session.exec(...)  # 加款
    return {"ok": True}
    # 正常返回 → get_transaction 的 yield 之后执行 commit
```

---

## 5. scope 参数——控制清理时机

### 5.1 默认行为：scope="request"

```python
def get_resource():
    resource = create_resource()
    try:
        yield resource
    finally:
        resource.close()    # 默认：在响应发送给客户端之后执行
```

默认情况下，yield 之后的清理代码在**响应发送给客户端之后**才执行。时间线：

```
请求到达 → 创建资源 → yield → 路径操作执行 → 返回响应 → 发送给客户端 → 清理资源
                                                                          ^^^^^^^^
                                                                    默认在这里清理
```

### 5.2 提前清理：scope="function"

```python
@app.get("/users/me")
def get_user_me(
    username: Annotated[str, Depends(get_username, scope="function")],
):
    return username
```

`scope="function"` 让清理代码在**路径操作函数返回之后、响应发送之前**执行：

```
请求到达 → 创建资源 → yield → 路径操作执行 → 清理资源 → 发送响应给客户端
                                               ^^^^^^^^
                                          提前到这里清理
```

### 5.3 什么时候用 scope="function"

```
scope="request"（默认）：
  资源在整个请求-响应周期内都可用
  适用于：数据库会话（可能在中间件或后台任务里还需要用）

scope="function"：
  资源在路径操作函数执行完就释放
  适用于：临时文件、锁、连接池中的稀缺连接
  好处是：资源更早释放，减少资源占用时间
```

### 5.4 子依赖项的 scope 规则

```
规则：父依赖项的 scope 不能比子依赖项"更早清理"

scope="request" 的依赖项 → 子依赖项也必须是 scope="request"
  因为子依赖项可能在清理阶段还要用

scope="function" 的依赖项 → 子依赖项可以是 "function" 或 "request"
  因为 function 比 request 先清理，子依赖项一定还在
```

用时间线看就清楚了：

```
                    创建          路径操作          函数结束       响应发送
                     │               │                │              │
scope="function":    ├───── 存活 ────┤── 清理 ──┤               │
scope="request":     ├───── 存活 ───────────────────┤── 清理 ──┤

如果父是 request，子是 function：
  父在"响应发送"后清理，此时想用子 → 但子在"函数结束"后就被清理了 → 错误！

如果父是 function，子是 request：
  父在"函数结束"后清理，此时子还在（要等"响应发送"后才清理）→ 没问题 ✓
```

**大多数情况下你不需要设置 scope**，默认的 `"request"` 就够了。这是一个高级特性。

---

## 6. yield 依赖项与上下文管理器的关系

### 6.1 Python 的上下文管理器（Context Manager）

```python
# 你每天都在用上下文管理器：
with open("file.txt") as f:
    content = f.read()
# f 自动关闭

with Session(engine) as session:
    heroes = session.exec(select(Hero)).all()
# session 自动关闭
```

`with` 语句背后是上下文管理器协议——`__enter__`（进入时执行）和 `__exit__`（退出时执行）。

### 6.2 yield 依赖项本质上就是上下文管理器

```python
# yield 依赖项
def get_session():
    session = Session(engine)
    try:
        yield session       # 相当于 __enter__ 返回值
    finally:
        session.close()     # 相当于 __exit__

# 等价的上下文管理器写法
class SessionManager:
    def __enter__(self):
        self.session = Session(engine)
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
```

FastAPI 在内部把 yield 依赖项包装成上下文管理器来使用。你不需要自己写 `@contextmanager` 装饰器——FastAPI 帮你做了。

### 6.3 在 yield 依赖项里使用上下文管理器

你可以把已有的上下文管理器直接嵌入 yield 依赖项：

```python
class MySuperContextManager:
    def __init__(self):
        self.db = DBSession()

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.close()


# 在 yield 依赖项里直接用 with
async def get_db():
    with MySuperContextManager() as db:
        yield db
    # with 退出时自动调用 __exit__，关闭数据库
```

这样你可以把任何第三方库提供的上下文管理器（文件操作、网络连接、锁、事务）无缝集成到 FastAPI 的依赖注入系统里。

---

## 7. 对应到你的投标项目——完整的依赖体系

把前面所有补充篇的知识整合到一起：

```python
# app/dependencies.py

import time
from typing import Annotated

from fastapi import Depends, Header, HTTPException
from sqlmodel import Session

from .database import engine


# ==================== yield 依赖项：数据库会话 ====================

def get_session():
    """每个请求一个数据库会话，请求结束后自动关闭"""
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]


# ==================== 子依赖项链：Token → 用户 → 管理员 ====================

async def get_token(authorization: Annotated[str, Header()]):
    """第一层：从请求头提取 Token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请提供有效的 Bearer Token")
    return authorization.removeprefix("Bearer ")


async def get_current_user(
    token: Annotated[str, Depends(get_token)],        # 子依赖项
    session: SessionDep,                                # yield 依赖项（缓存共享）
):
    """第二层：解析 Token 获取用户"""
    # 实际项目用 JWT 解码
    user = {"username": "alice", "role": "admin"}       # 模拟
    if not user:
        raise HTTPException(status_code=401, detail="Token 无效")
    return user

UserDep = Annotated[dict, Depends(get_current_user)]


async def require_admin(user: UserDep):
    """第三层：检查管理员权限（装饰器依赖用，不需要返回值也行）"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


# ==================== 全局依赖项：请求日志 ====================

async def log_request():
    """记录每个请求（全局依赖项，不需要返回值）"""
    print(f"[{time.strftime('%H:%M:%S')}] 收到请求")
```

```python
# app/main.py

from fastapi import Depends, FastAPI
from .dependencies import log_request, require_admin
from .routers import projects
from .internal import admin

# 全局依赖项：所有接口都记录日志
app = FastAPI(
    title="智能投标决策系统",
    dependencies=[Depends(log_request)],
)

# 项目管理：不需要额外的 Router 级依赖（认证在接口参数里按需添加）
app.include_router(projects.router, prefix="/api")

# 管理后台：整个模块需要管理员权限（装饰器依赖）
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["管理后台"],
    dependencies=[Depends(require_admin)],
)
```

完整的依赖树和执行顺序：

```
请求到达 POST /admin/clear-cache

① 全局依赖项
   log_request() → 记录日志

② Router 级依赖项（include_router 的 dependencies）
   require_admin()
     → get_current_user()              ← 子依赖项
         → get_token()                 ← 子子依赖项（提取 Authorization 头）
         → get_session()               ← yield 依赖项（创建数据库会话）
     → 检查 role == "admin"

③ 路径操作函数执行
   clear_cache() → 返回结果

④ 响应发送给客户端

⑤ yield 依赖项的清理代码执行
   get_session 的 with 块退出 → session.close()
```

---

## 8. 本节核心知识点速查

```
概念                                    做法
──────────────────────────────────────────────────────────────────────────
全局依赖项                              FastAPI(dependencies=[Depends(func)])
yield 依赖项                            try: yield resource  finally: cleanup()
yield + with                            with Session(engine) as s: yield s
清理保证执行                            用 try/finally
捕获异常并转换                          except XError: raise HTTPException(...)
捕获异常并重新抛出                       except XError: log(); raise
绝对不要吞异常                           except 里必须 raise 或 raise HTTPException
清理顺序                                和创建顺序相反（后创建的先清理）
scope="request"（默认）                  响应发送后清理
scope="function"                        函数返回后、响应发送前清理
上下文管理器集成                         在 yield 依赖项里用 with
```

**本节最核心的三句话**：

1. **全局依赖项就是加在 `FastAPI()` 上的装饰器依赖**。一行 `FastAPI(dependencies=[...])` 让所有接口都经过这些依赖项的验证。和 Router 级、路径操作级依赖项叠加使用，形成多层"安检"：全局 → Router → 路径操作 → 参数。

2. **yield 依赖项的核心模式是 `try: yield resource  finally: cleanup`**。yield 之前创建资源，yield 交出资源给路径操作函数用，yield 之后（在 finally 里）清理资源。`finally` 保证无论请求成功还是异常，清理代码都会执行。这就是数据库会话、文件句柄、锁等资源的标准管理方式。

3. **多层 yield 依赖项的清理顺序和创建顺序相反**。先创建的最后清理——就像嵌套的 `with` 块。FastAPI 自动管理这个顺序，清理 C 的时候 B 还在，清理 B 的时候 A 还在，你可以放心地在清理代码里使用子依赖项提供的资源。
