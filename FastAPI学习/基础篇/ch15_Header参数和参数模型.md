上一节学了 Cookie，这一节学 Header。学完之后，FastAPI 的五种数据来源（路径、查询参数、请求体、Cookie、Header）就**全部掌握了**



Header 的基本语法和 Cookie 一模一样，但有一个独特的坑：**HTTP 请求头用连字符（`-`），Python 变量用下划线（`_`），FastAPI 会自动帮你转换**



HTTP相关内容在预备知识里，可以回顾

---

## 1. 先搞懂 HTTP Header 是什么

### 1.1 每个 HTTP 请求都自带一堆 Header

当浏览器发请求时，除了 URL 和请求体之外，还会自动附带一些"元信息"，放在请求头（Header）里：

```css
GET /items/ HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...
Accept: application/json
Accept-Language: zh-CN,zh;q=0.9
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
X-Request-ID: a3b2c1d4-e5f6-7890
```

每一行就是一个 Header，格式是 `名字: 值`

### 1.2 Header 和 Cookie 的区别

```css
Cookie：
  - 浏览器自动管理（设一次，之后每次请求自动带上）
  - 主要用来存"状态"（登录信息、用户偏好）
  - 有大小限制（通常 4KB）

Header：
  - 每次请求单独设置
  - 用来传"元信息"（认证令牌、内容类型、客户端信息、自定义标记）
  - 没有严格大小限制
```

### 1.3 常见的 Header 有哪些

```css
浏览器自动发的：
  User-Agent        → 客户端信息（浏览器类型、操作系统）
  Accept            → 客户端能接受的数据格式
  Accept-Language   → 客户端的语言偏好
  Host              → 请求的域名

开发者手动设的：
  Authorization     → 认证令牌（JWT Token 等）
  X-Request-ID      → 请求追踪 ID（自定义 Header 通常以 X- 开头）
  X-API-Key         → API 密钥
  Content-Type      → 请求体的格式（application/json 等）
```

---

## 2. 基本用法：读取单个 Header

### 2.1 语法

```python
from typing import Annotated
from fastapi import FastAPI, Header

app = FastAPI()


@app.get("/items/")
def read_items(
    user_agent: Annotated[str | None, Header()] = None,
):
    """
    读取请求头里的 User-Agent。

    客户端发送的 Header 名是：User-Agent（连字符，大小写不敏感）
    你在代码里写的参数名是：user_agent（下划线，小写）
    FastAPI 自动帮你做了转换。
    """
    return {"User-Agent": user_agent}
```

### 2.2 和其他四种参数的对比

```python
from fastapi import Path, Query, Body, Cookie, Header

# 从 URL 路径取
item_id: Annotated[int, Path()]

# 从 URL 查询参数取
page: Annotated[int, Query()] = 1

# 从请求体取
data: Annotated[Item, Body()]

# 从 Cookie 取
session_id: Annotated[str | None, Cookie()] = None

# 从 Header 取
user_agent: Annotated[str | None, Header()] = None
```

五种写法的模式完全一样，换个标记函数就行

---

## 3. Header 独有的特性：下划线自动转连字符

### 3.1 问题

HTTP 标准里，Header 名字用**连字符**分隔单词：

```css
User-Agent
Accept-Language
X-Request-ID
If-Modified-Since
```

但 Python 变量名**不能有连字符**（连字符会被当成减号）：

```python
# ❌ Python 语法错误：这是减法运算，不是变量名
user-agent = "Mozilla"

# ✅ Python 变量只能用下划线
user_agent = "Mozilla"
```

### 3.2 FastAPI 的自动转换

FastAPI 帮你解决了这个矛盾——**你在代码里用下划线，FastAPI 自动把它转成连字符去匹配 Header**：

```python
@app.get("/items/")
def read_items(
    user_agent: Annotated[str | None, Header()] = None,
    accept_language: Annotated[str | None, Header()] = None,
    x_request_id: Annotated[str | None, Header()] = None,
):
    """
    你写的参数名           FastAPI 实际去匹配的 Header 名
    ─────────────────────────────────────────────────
    user_agent        →    User-Agent
    accept_language   →    Accept-Language
    x_request_id      →    X-Request-ID
    """
    return {
        "User-Agent": user_agent,
        "Accept-Language": accept_language,
        "X-Request-ID": x_request_id,
    }
```

转换规则很简单：**下划线 `_` → 连字符 `-`**。大小写无所谓，HTTP Header 本来就不区分大小写

### 3.3 如果你的 Header 真的用下划线呢

极少数情况下，某些非标准的 Header 名字里确实用了下划线（比如一些老系统）。这时候你不希望 FastAPI 做转换：

```python
@app.get("/items/")
def read_items(
    # 关闭自动转换：参数名 strange_header 就直接匹配 Header 名 strange_header
    strange_header: Annotated[
        str | None,
        Header(convert_underscores=False),
    ] = None,
):
    return {"strange_header": strange_header}
```

`convert_underscores=False` 告诉 FastAPI："别转换，我的 Header 名字里就是下划线。"**但一般别这么做**，很多 HTTP 代理和服务器会丢弃带下划线的 Header，你的请求可能根本到不了你的服务

---

## 4. 重复的 Header：同一个名字出现多次

### 4.1 场景

HTTP 协议允许同一个 Header 出现多次。比如客户端可能同时发送：

```
X-Token: token-aaa
X-Token: token-bbb
```

### 4.2 怎么接收

把类型声明为 `list[str]`，FastAPI 就会把同名 Header 的所有值收集成一个列表：

```python
from typing import Annotated
from fastapi import FastAPI, Header

app = FastAPI()


@app.get("/items/")
def read_items(
    x_token: Annotated[list[str] | None, Header()] = None,
):
    """
    客户端发送：
      X-Token: token-aaa
      X-Token: token-bbb

    函数收到：
      x_token = ["token-bbb", "token-aaa"]
    """
    return {"X-Token values": x_token}
```

### 4.3 什么时候会遇到重复 Header

```
常见场景：
  - 多个认证令牌（多层代理各加一个）
  - 多个转发地址（X-Forwarded-For 经过多个代理时）
  - 自定义的多值 Header

大部分时候用不到，知道有这个功能就行
```

---

## 5. 多个 Header 打包成模型

### 5.1 语法

和查询参数模型、Cookie 模型完全一样的套路：

```python
from typing import Annotated
from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI()


class CommonHeaders(BaseModel):
    """常用的请求头——打包成模型"""
    host: str                              # 必填
    save_data: bool                        # 布尔类型：客户端是否要求节省流量
    if_modified_since: str | None = None   # 可选：条件请求
    traceparent: str | None = None         # 可选：分布式追踪
    x_tag: list[str] = []                  # 可重复的自定义 Header


@app.get("/items/")
def read_items(headers: Annotated[CommonHeaders, Header()]):
    return headers
```

### 5.2 模型里的字段名也会自动转换

模型里的字段名同样遵循"下划线 → 连字符"的转换规则：

```python
class CommonHeaders(BaseModel):
    host: str                    # → 匹配 Header: Host
    save_data: bool              # → 匹配 Header: Save-Data
    if_modified_since: str       # → 匹配 Header: If-Modified-Since
    x_tag: list[str] = []       # → 匹配 Header: X-Tag
```

你在模型里写 Python 风格的下划线变量名，FastAPI 自动去匹配连字符风格的 HTTP Header 名

### 5.3 模型也能禁止多余的 Header

```python
class CommonHeaders(BaseModel):
    model_config = {"extra": "forbid"}    # 禁止多余的 Header

    host: str
    save_data: bool
    if_modified_since: str | None = None
    traceparent: str | None = None
    x_tag: list[str] = []
```

**实际建议：Header 上几乎不要用 forbid。** 浏览器和各种中间件会自动加很多 Header（`User-Agent`、`Accept`、`Connection` 等），forbid 会导致这些全部触发报错。除非你完全控制客户端（比如微服务之间的内部通信），否则别用

### 5.4 模型也能关闭下划线转换

```python
@app.get("/items/")
def read_items(
    headers: Annotated[CommonHeaders, Header(convert_underscores=False)],
):
    # 模型里的 save_data 直接匹配 Header 名 save_data（不转成 save-data）
    return headers
```

---

## 6. 一个完整的实际例子：认证 + 追踪 + 限流

```python
from typing import Annotated
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()


@app.get("/api/protected-data/")
def get_protected_data(
    # 认证令牌——几乎每个需要登录的接口都会用
    authorization: Annotated[str, Header(
        description="Bearer Token，格式：Bearer eyJhbG..."
    )],

    # 请求追踪 ID——方便排查问题
    x_request_id: Annotated[str | None, Header(
        description="请求追踪 ID，用于日志关联"
    )] = None,

    # 客户端信息
    user_agent: Annotated[str | None, Header()] = None,
):
    """
    客户端需要发送：
      Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
      X-Request-ID: a3b2c1d4-e5f6-7890      （可选）
      User-Agent: MyApp/1.0                   （浏览器自动发）

    代码里对应的参数名：
      authorization  → 匹配 Header: Authorization
      x_request_id   → 匹配 Header: X-Request-ID
      user_agent     → 匹配 Header: User-Agent
    """
    # 检查令牌格式
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="令牌格式错误")

    token = authorization.removeprefix("Bearer ")

    return {
        "token_preview": token[:20] + "...",
        "request_id": x_request_id,
        "client": user_agent,
    }
```

**这个例子展示了 Header 最常见的用法**：`Authorization` 做认证，`X-Request-ID` 做追踪，`User-Agent` 获取客户端信息。

---

## 7. 五种数据来源的最终总结

到这一节为止，FastAPI 的五种数据来源全部学完了。放在一起做个最终对比：

### 7.1 五种来源一览

```
数据来源          标记方式      来源位置              自动推断？    独特功能
──────────────────────────────────────────────────────────────────────────────
URL 路径         Path()       /items/{id}           ✅ 自动       —
URL 查询参数     Query()      /items?page=2         ✅ 自动       —
HTTP 请求体      Body()       JSON Body             ✅ 自动       embed=True
HTTP Cookie      Cookie()     Cookie 请求头          ❌ 必须标记   —
HTTP Header      Header()     各种请求头             ❌ 必须标记   下划线转连字符
```

### 7.2 参数判断规则完整版（终极版）

```
FastAPI 看到一个函数参数，判断它从哪里取数据：

1. 参数名出现在路径 {} 里                       → 路径参数
2. 参数被 Annotated[..., Path()] 标记           → 路径参数
3. 参数被 Annotated[..., Body()] 标记           → 请求体
4. 参数被 Annotated[..., Query()] 标记          → 查询参数
5. 参数被 Annotated[..., Cookie()] 标记         → Cookie
6. 参数被 Annotated[..., Header()] 标记         → Header
7. 参数类型是 Pydantic 模型（且没标记来源）       → 请求体
8. 以上都不是，且是简单类型                       → 查询参数（默认）
```

### 7.3 全部都支持的通用功能

```python
# 所有来源都能加约束
Annotated[int, Query(ge=1, le=100)]
Annotated[str, Header(min_length=10)]
Annotated[str, Cookie(max_length=64)]

# 所有来源都能加文档信息
Annotated[str, Header(description="认证令牌", examples=["Bearer xxx"])]

# 所有来源都能打包成 Pydantic 模型
filters: Annotated[FilterParams, Query()]       # 查询参数模型
cookies: Annotated[MyCookies, Cookie()]         # Cookie 模型
headers: Annotated[CommonHeaders, Header()]     # Header 模型
```

---

## 8. 对应到你的投标项目

```python
from typing import Annotated
from fastapi import FastAPI, Path, Query, Header, Cookie, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="智能投标决策系统")


# ==================== Header 模型 ====================

class AuthHeaders(BaseModel):
    """认证相关的 Header"""
    authorization: str = Field(description="Bearer Token")
    x_request_id: str | None = Field(
        default=None,
        description="请求追踪 ID"
    )
    x_client_version: str | None = Field(
        default=None,
        description="客户端版本号"
    )


# ==================== 五种数据来源全部上场 ====================

@app.get("/api/projects/{project_id}/bids")
def get_project_bids(
    # ① 路径参数：从 URL 路径取
    project_id: Annotated[int, Path(ge=1, description="项目 ID")],

    # ② 查询参数：从 URL ?key=value 取
    page: Annotated[int, Query(ge=1, description="页码")] = 1,
    sort_by: Annotated[str, Query(description="排序字段")] = "created_at",

    # ③ Header：从 HTTP 请求头取（认证信息打包成模型）
    auth: Annotated[AuthHeaders, Header()],

    # ④ Cookie：从 Cookie 取
    theme: Annotated[str, Cookie(description="界面主题")] = "light",
):
    """
    一个接口同时使用了四种数据来源：

    请求示例：
    GET /api/projects/42/bids?page=2&sort_by=bid_price
    Authorization: Bearer eyJhbG...
    X-Request-ID: req-abc-123
    Cookie: theme=dark

    数据来源对照：
      project_id = 42                    ← URL 路径
      page = 2, sort_by = "bid_price"    ← URL 查询参数
      auth.authorization = "Bearer ..."  ← HTTP Header
      auth.x_request_id = "req-abc-123"  ← HTTP Header
      theme = "dark"                     ← Cookie
    """
    # 验证令牌
    if not auth.authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="令牌格式错误")

    return {
        "project_id": project_id,
        "page": page,
        "sort_by": sort_by,
        "request_id": auth.x_request_id,
        "client_version": auth.x_client_version,
        "theme": theme,
        "results": ["...投标记录列表..."],
    }


# ==================== 更简单的用法：单独的 Header 参数 ====================

@app.get("/api/health")
def health_check(
    # 直接读单个 Header，不用模型
    x_forwarded_for: Annotated[str | None, Header(
        description="客户端真实 IP（经过代理时）"
    )] = None,
    user_agent: Annotated[str | None, Header()] = None,
):
    """
    健康检查接口：

    x_forwarded_for → 匹配 Header: X-Forwarded-For
    user_agent      → 匹配 Header: User-Agent
    """
    return {
        "status": "healthy",
        "client_ip": x_forwarded_for,
        "client": user_agent,
    }
```

---

## 9. 本节核心知识点速查

```css
概念                               写法                               说明
───────────────────────────────────────────────────────────────────────────────
读取单个 Header        Annotated[str | None, Header()] = None       从请求头取值
下划线自动转连字符      user_agent → 匹配 User-Agent                   默认开启
关闭自动转换           Header(convert_underscores=False)             极少使用
重复 Header            Annotated[list[str] | None, Header()]        同名 Header 多个值
Header 打包成模型      Annotated[CommonHeaders, Header()]            多个 Header 统一管理
```

**本节最核心的两句话**：

1. `Header()` 和 `Query()`、`Path()`、`Cookie()` 是一家人，语法完全一样，只是告诉 FastAPI "去 HTTP 请求头里取数据"

2. Header 有一个独特的自动转换：你在代码里写 `user_agent`（下划线），FastAPI 自动去匹配 `User-Agent`（连字符）。因为 HTTP Header 用连字符，但 Python 变量名不能用连字符























