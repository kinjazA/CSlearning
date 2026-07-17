前面学了从 URL 路径取数据（Path）、从 URL 查询参数取数据（Query）、从请求体取数据（Body）。但 HTTP 请求里还有一个地方可以携带数据——**Cookie**

用户的登录状态、偏好设置、追踪信息，通常都存在 Cookie 里。这一节讲的就是：**怎么在 FastAPI 里读取 Cookie 的值。**好消息是：语法和 Query、Path 完全一样，你学过的东西直接套用



==FastAPI 里的 `Header()` 和 `Cookie()` 参数名不是随便写的，FastAPI 会根据参数名去请求头或 Cookie 里找对应字段，比如 `user_agent` 会对应 `User-Agent`==

---

## 1. 先搞懂 Cookie 是什么

### 1.1 Cookie 的本质

Cookie 是浏览器自动存储和发送的一小段数据。工作流程是这样的：

```css
第一步：服务器在响应里说"帮我存一下这个值"
  ← HTTP 响应头：Set-Cookie: session_id=abc123

第二步：浏览器把这个值存起来

第三步：以后每次请求同一个网站，浏览器自动把 Cookie 带上
  → HTTP 请求头：Cookie: session_id=abc123
```

**关键点**：Cookie 是浏览器**自动**带上的，不需要前端手动操作。用户登录后，服务器设一个 `session_id` Cookie，之后每次请求浏览器都会自动带上，服务器就能认出"这是已登录用户"

### 1.2 Cookie vs 查询参数 vs 请求体

```css
查询参数  → 明文在 URL 里，用户看得见：/items?page=2&sort=price
请求体    → 在 HTTP Body 里，适合传大量数据：JSON、表单
Cookie   → 在 HTTP 请求头里，浏览器自动管理，适合存"状态信息"
```

```css
什么数据放在 Cookie 里？
  - 登录凭证（session_id、token）
  - 用户偏好（语言、主题）
  - 追踪信息（广告 ID、分析标记）

什么数据不放在 Cookie 里？
  - 搜索关键词、分页参数 → 放查询参数
  - 表单内容、大段数据 → 放请求体
```

---

## 2. 基本用法：读取单个 Cookie

### 2.1 语法

```python
from typing import Annotated
from fastapi import Cookie, FastAPI

app = FastAPI()


@app.get("/items/")
def read_items(
    ads_id: Annotated[str | None, Cookie()] = None,
):
    """
    从请求的 Cookie 里读取 ads_id 的值。
    如果浏览器没有这个 Cookie，ads_id 就是 None。
    """
    return {"ads_id": ads_id}
```

### 2.2 和 Query、Path 的对比

```python
from fastapi import Query, Path, Cookie

# 从 URL 路径取 → Path()
item_id: Annotated[int, Path()]

# 从 URL 查询参数取 → Query()
page: Annotated[int, Query()] = 1

# 从 Cookie 取 → Cookie()
session_id: Annotated[str | None, Cookie()] = None
```

**三者写法完全一样**，只是把 `Query()` / `Path()` 换成了 `Cookie()`。所有的约束参数（`gt`、`max_length`、`description` 等）在 `Cookie()` 里也能用

### 2.3 为什么必须写 Cookie()

```python
# ❌ 不写 Cookie()，FastAPI 会把它当成查询参数
def read_items(ads_id: str | None = None):
    # FastAPI 认为 ads_id 是查询参数，从 URL ?ads_id=xxx 取值
    ...

# ✅ 写了 Cookie()，FastAPI 知道从 Cookie 里取
def read_items(ads_id: Annotated[str | None, Cookie()] = None):
    # FastAPI 从 HTTP 请求头的 Cookie 里取 ads_id 的值
    ...
```

回忆参数判断规则：简单类型默认是查询参数。**必须显式用 `Cookie()` 标记，FastAPI 才会去 Cookie 里找**

---

## 3. Cookie 也支持约束

`Cookie()` 和 `Query()` 是一家人，支持一样的约束：

```python
from typing import Annotated
from fastapi import Cookie, FastAPI

app = FastAPI()


@app.get("/dashboard/")
def read_dashboard(
    # 必填的 Cookie（没有默认值 → 必填）
    session_id: Annotated[str, Cookie(
        min_length=32,
        max_length=64,
        description="登录会话 ID",
    )],

    # 可选的 Cookie（有默认值 → 可选）
    theme: Annotated[str, Cookie(
        description="用户界面主题",
    )] = "light",

    # 可选的 Cookie
    language: Annotated[str | None, Cookie(
        description="用户语言偏好",
    )] = None,
):
    return {
        "session_id": session_id,
        "theme": theme,
        "language": language,
    }
```

---

## 4. 多个 Cookie 打包成模型

### 4.1 问题：Cookie 多了函数签名又变长了

```python
# Cookie 越来越多，函数签名越来越长
@app.get("/items/")
def read_items(
    session_id: Annotated[str, Cookie()],
    csrf_token: Annotated[str | None, Cookie()] = None,
    theme: Annotated[str, Cookie()] = "light",
    language: Annotated[str | None, Cookie()] = None,
    tracking_id: Annotated[str | None, Cookie()] = None,
):
    ...
```

这个问题你在第七节学查询参数模型时见过——解决方案也一样：**打包成 Pydantic 模型。**

### 4.2 用模型打包 Cookie

```python
from typing import Annotated
from fastapi import Cookie, FastAPI
from pydantic import BaseModel

app = FastAPI()


class MyCookies(BaseModel):
    """把一组相关的 Cookie 打包成模型"""
    session_id: str
    fatebook_tracker: str | None = None
    googall_tracker: str | None = None


@app.get("/items/")
def read_items(cookies: Annotated[MyCookies, Cookie()]):
    """
    FastAPI 自动从请求的 Cookie 里提取
    session_id、fatebook_tracker、googall_tracker 的值，
    填进 MyCookies 模型里。
    """
    return cookies
```

**语法规律和查询参数模型完全一样**：

```python
# 查询参数模型（第七节学过）
filter_query: Annotated[FilterParams, Query()]
#                        ↑ 模型          ↑ 从查询参数取

# Cookie 模型（这一节）
cookies: Annotated[MyCookies, Cookie()]
#                   ↑ 模型       ↑ 从 Cookie 取
```

就是把 `Query()` 换成了 `Cookie()`，其他一切都一样

### 4.3 在函数里怎么用

```python
@app.get("/items/")
def read_items(cookies: Annotated[MyCookies, Cookie()]):
    # cookies 是一个 MyCookies 对象，用点号访问属性
    print(cookies.session_id)           # "abc123"
    print(cookies.fatebook_tracker)     # "track_xyz" 或 None
    print(cookies.googall_tracker)      # "ga_456" 或 None

    # 转成字典
    print(cookies.model_dump())
    # {"session_id": "abc123", "fatebook_tracker": "track_xyz", ...}

    return cookies
```

---

## 5. 禁止多余的 Cookie

和查询参数模型一样，可以用 `model_config = {"extra": "forbid"}` 禁止客户端发送模型里没定义的 Cookie：

```python
from typing import Annotated
from fastapi import Cookie, FastAPI
from pydantic import BaseModel

app = FastAPI()


class MyCookies(BaseModel):
    model_config = {"extra": "forbid"}   # 禁止多余的 Cookie

    session_id: str
    fatebook_tracker: str | None = None
    googall_tracker: str | None = None


@app.get("/items/")
def read_items(cookies: Annotated[MyCookies, Cookie()]):
    return cookies
```

如果客户端发了一个 `santa_tracker` Cookie（模型里没定义的）：

```json
{
    "detail": [
        {
            "type": "extra_forbidden",
            "loc": ["cookie", "santa_tracker"],
            "msg": "Extra inputs are not permitted",
            "input": "good-list-please"
        }
    ]
}
```

**实际用不用 forbid**：大部分情况**不用**。浏览器会自动发送很多 Cookie（来自各种网站和第三方服务），如果 forbid 了，这些自动带上的 Cookie 都会触发报错。只有在非常严格的安全场景下才考虑使用

---

## 6. Cookie 的特殊之处：在 /docs 里测不了

### 6.1 为什么

打开 `/docs`，你能看到 Cookie 参数的文档，但**点 "Try it out" 测试时 Cookie 不会被发送**。

这是因为 `/docs` 页面（Swagger UI）是用 JavaScript 运行的，而浏览器出于安全原因限制了 JavaScript 对 Cookie 的访问。Swagger UI 没法帮你把 Cookie 塞进请求里

### 6.2 怎么测试 Cookie

```css
方法一：用 curl 命令行
  curl -X GET http://localhost:8000/items/ \
    --cookie "session_id=abc123;theme=dark"

方法二：用 Python 的 requests 库
  import requests
  r = requests.get(
      "http://localhost:8000/items/",
      cookies={"session_id": "abc123", "theme": "dark"}
  )

方法三：用 Postman 或 Insomnia 等 API 测试工具
  在请求的 Cookies 标签页里手动添加

方法四：用浏览器开发者工具
  先在浏览器控制台设置 Cookie：document.cookie = "session_id=abc123"
  然后访问接口
```

---

## 7. 完整的参数判断规则

加上 Cookie 之后，完整版是这样的：

```css
FastAPI 看到一个函数参数，判断它从哪里取数据：

1. 参数名在路径 {} 里                         → 路径参数
2. 参数类型是 Pydantic 模型（且没标记来源）     → 请求体
3. Annotated[..., Body()] 标记                → 请求体
4. Annotated[..., Query()] 标记               → 查询参数
5. Annotated[..., Cookie()] 标记              → Cookie
6. Annotated[..., Header()] 标记              → 请求头（下一节学）
7. 以上都不是，且是简单类型                     → 查询参数（默认）
```

**Cookie 和 Header 必须显式标记**，FastAPI 不会自动猜。不标记的简单类型默认是查询参数

---

## 8. 对应到你的投标项目

```python
from typing import Annotated
from fastapi import FastAPI, Cookie, Query
from pydantic import BaseModel, Field

app = FastAPI(title="智能投标决策系统")


# ==================== Cookie 模型 ====================

class UserSession(BaseModel):
    """用户会话信息——从 Cookie 中读取"""
    session_id: str = Field(
        min_length=32,
        description="登录会话 ID"
    )
    user_role: str = Field(
        default="viewer",
        description="用户角色：admin / editor / viewer"
    )
    preferred_strategy: str = Field(
        default="moderate",
        description="用户偏好的默认投标策略"
    )


# ==================== 结合查询参数和 Cookie 使用 ====================

@app.get("/api/bids/")
def list_bids(
    # 从 Cookie 取用户会话信息
    session: Annotated[UserSession, Cookie()],

    # 从查询参数取搜索条件
    page: Annotated[int, Query(ge=1)] = 1,
    keyword: Annotated[str | None, Query()] = None,
):
    """
    请求示例：
    GET /api/bids/?page=2&keyword=路面
    Cookie: session_id=a1b2c3d4...; user_role=editor; preferred_strategy=aggressive

    session 从 Cookie 取，page 和 keyword 从 URL 查询参数取。
    """
    return {
        "user_role": session.user_role,
        "strategy": session.preferred_strategy,
        "page": page,
        "keyword": keyword,
        "results": ["..."],
    }


# ==================== 单独使用 Cookie 参数 ====================

@app.get("/api/user/preferences")
def get_preferences(
    theme: Annotated[str, Cookie(description="界面主题")] = "light",
    language: Annotated[str, Cookie(description="语言偏好")] = "zh-CN",
    dashboard_layout: Annotated[str, Cookie(description="仪表盘布局")] = "default",
):
    """
    读取用户的偏好设置。
    这些设置存在 Cookie 里，用户切换主题或语言时由前端写入 Cookie，
    后端每次请求都能读到。
    """
    return {
        "theme": theme,
        "language": language,
        "dashboard_layout": dashboard_layout,
    }
```

---

## 9. Query / Path / Body / Cookie 大一统

到这一节为止，已经学了四种数据来源。它们的使用方式高度统一：

```
数据来源         标记方式         导入位置          默认推断
──────────────────────────────────────────────────────────────
URL 路径        Path()          from fastapi      参数名在 {} 里 → 自动推断
URL 查询参数    Query()         from fastapi      简单类型 → 自动推断
HTTP 请求体     Body()          from fastapi      Pydantic 模型 → 自动推断
HTTP Cookie     Cookie()        from fastapi      ❌ 不会自动推断，必须显式标记
HTTP Header     Header()        from fastapi      ❌ 不会自动推断，必须显式标记（下节学）
```

**约束参数全部通用**：`gt`、`ge`、`max_length`、`description`、`examples` 等所有约束参数，在 Path / Query / Body / Cookie / Header 里都能用，语法完全一样

**模型打包全部通用**：

```python
filter_query: Annotated[FilterParams, Query()]     # 查询参数打包成模型
cookies:      Annotated[MyCookies,    Cookie()]    # Cookie 打包成模型
# headers:   Annotated[MyHeaders,    Header()]    # Header 打包成模型（下节学）
```

---

## 10. 本节核心知识点速查

```css
概念                  写法                                      效果
───────────────────────────────────────────────────────────────────────────
读取单个 Cookie      Annotated[str | None, Cookie()] = None   从 Cookie 取值
Cookie 加约束        Cookie(min_length=32, description="...")  校验 + 文档
Cookie 打包成模型    Annotated[MyCookies, Cookie()]           多个 Cookie 统一管理
禁止多余 Cookie      model_config = {"extra": "forbid"}       多传就报错
```

**本节最核心的一句话**：`Cookie()` 和 `Query()`、`Path()` 是一家人，语法完全一样，只是告诉 FastAPI "去 Cookie 里取数据"。简单类型默认是查询参数，必须显式写 `Cookie()` 才会从 Cookie 取。Cookie 也能打包成 Pydantic 模型统一管理

