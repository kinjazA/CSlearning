**路径操作配置**——给接口加标签、摘要、描述、弃用标记等元数据，让 `/docs` 文档更专业



**JSON 兼容编码器**——把 Pydantic 模型、datetime 等 Python 对象转成 JSON 能表示的格式

---

# 第一部分：路径操作配置

## 1. 所有配置参数一览

路径操作配置写在**装饰器**里，不是函数参数里：

```python
@app.post(
    "/items/",                              # 路径
    response_model=Item,                    # 响应模型
    status_code=status.HTTP_201_CREATED,    # 状态码
    tags=["items"],                         # 标签 
    summary="创建商品",                      # 摘要 
    description="创建一个商品，包含名称...",    # 描述
    response_description="创建成功的商品",    # 响应描述
    deprecated=False,                       # 是否弃用
)
def create_item(item: Item):
    return item
```

前面学过的 `response_model` 和 `status_code` 也是路径操作配置的一部分，它们都写在装饰器里

---

## 2. `tags`——给接口分组

### 2.1 问题

当你的 API 有几十个接口时，`/docs` 页面会变成一个巨长的列表，根本找不到想要的接口。

### 2.2 用标签分组

```python
from fastapi import FastAPI

app = FastAPI()


@app.post("/items/", tags=["项目管理"])
def create_item():
    return {"name": "Foo"}


@app.get("/items/", tags=["项目管理"])
def read_items():
    return [{"name": "Foo"}]


@app.get("/users/", tags=["用户管理"])
def read_users():
    return [{"username": "alice"}]


@app.post("/bids/", tags=["投标管理"])
def create_bid():
    return {"bid_id": "001"}
```

效果：`/docs` 页面上接口会按标签分组展示，每组有一个折叠标题：

```
▼ 项目管理
    POST /items/
    GET  /items/
▼ 用户管理
    GET  /users/
▼ 投标管理
    POST /bids/
```

### 2.3 一个接口可以有多个标签

```python
@app.get("/projects/{project_id}/bids", tags=["项目管理", "投标管理"])
def get_project_bids(project_id: int):
    ...
```

这个接口会同时出现在"项目管理"和"投标管理"两个分组下

### 2.4 用枚举管理标签（大项目推荐）

当项目大了，标签散落在各个文件里，容易写错（一个写了"项目管理"，另一个写了"项目"）。用枚举统一管理：

```python
from enum import Enum


class Tags(Enum):
    projects = "项目管理"
    users = "用户管理"
    bids = "投标管理"
    system = "系统管理"


@app.post("/items/", tags=[Tags.projects])
def create_item():
    ...

@app.get("/users/", tags=[Tags.users])
def read_users():
    ...
```

好处：

- 编辑器自动补全，不会写错
- 改标签名字只改枚举一个地方
- 所有标签集中管理，一目了然

---

## 3. `summary` 和 `description`——给接口加说明

### 3.1 `summary`——一句话概括

```python
@app.post("/items/", summary="创建商品")
def create_item(item: Item):
    ...
```

`summary` 显示在 `/docs` 页面接口标题旁边，一般不超过一行。

**不写 summary 的话**，FastAPI 会用函数名生成：`create_item` → "Create Item"。

### 3.2 `description`——详细说明

有两种写法：

**写法一**：在装饰器里写（适合短描述）

```python
@app.post(
    "/items/",
    summary="创建商品",
    description="创建一个商品，需要提供名称、价格等信息。价格必须大于 0。",
)
def create_item(item: Item):
    ...
```

**写法二**：用函数的 docstring 写（推荐，适合长描述，支持 Markdown）

```python
@app.post("/items/", summary="创建商品")
def create_item(item: Item):
    """
    创建一个商品，需要提供以下信息：

    - **name**：商品名称（必填）
    - **description**：详细描述（可选）
    - **price**：价格，必须大于 0（必填）
    - **tax**：税费（可选，默认无税）
    - **tags**：标签集合，自动去重（可选）
    """
    return item
```

**docstring 里可以写 Markdown**——加粗、列表、代码块都支持，`/docs` 页面会正确渲染。

**如果同时写了 `description` 参数和 docstring**，`description` 参数优先。所以推荐只用 docstring，不用 `description` 参数

### 3.3 `response_description`——描述响应

```python
@app.post(
    "/items/",
    summary="创建商品",
    response_description="创建成功后返回的商品对象",
)
def create_item(item: Item):
    """创建一个商品..."""
    return item
```

注意区分：

```
description / docstring    → 描述这个接口"做什么"（整体说明）
response_description       → 描述这个接口"返回什么"（响应说明）
```

不写 `response_description` 的话，FastAPI 默认生成 "Successful response"

---

## 4. `deprecated`——标记接口为已弃用

### 4.1 场景

老接口要下线，但客户端还在用，不能直接删。先标记为"弃用"，给客户端时间迁移

### 4.2 用法

```python
# 新接口
@app.get("/api/v2/items/", tags=["项目管理"])
def read_items_v2():
    return [{"name": "Foo", "version": 2}]


# 老接口：标记为弃用
@app.get("/api/v1/items/", tags=["项目管理"], deprecated=True)
def read_items_v1():
    """
    ⚠️ 此接口已弃用，请使用 /api/v2/items/

    将在 2025 年 12 月 31 日后下线。
    """
    return [{"name": "Foo", "version": 1}]
```

效果：`/docs` 页面上这个接口会被画上删除线，颜色变灰，提醒开发者别用了。**接口本身还能正常调用**——`deprecated=True` 只影响文档显示，不影响功能

---

## 5. 对应到你的投标项目

```python
from enum import Enum

from fastapi import FastAPI, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="智能投标决策系统",
    description="提供项目管理、投标管理、数据分析等 API 服务",
    version="2.0.0",
)


# ==================== 标签枚举 ====================

class Tags(Enum):
    projects = "项目管理"
    bids = "投标管理"
    users = "用户管理"
    analysis = "数据分析"
    system = "系统管理"


# ==================== 项目接口 ====================

@app.post(
    "/api/projects/",
    tags=[Tags.projects],
    summary="创建项目",
    status_code=status.HTTP_201_CREATED,
    response_description="创建成功的项目信息",
)
def create_project():
    """
    创建一个新的工程项目，需要提供以下信息：

    - **project_name**：项目名称
    - **location**：项目所在地
    - **control_price**：控制价（万元）
    - **bid_deadline**：投标截止时间
    """
    return {"project_name": "示例项目"}


@app.get(
    "/api/projects/",
    tags=[Tags.projects],
    summary="查询项目列表",
)
def list_projects():
    """
    分页查询项目列表，支持按名称、地区、状态筛选。
    """
    return []


# ==================== 投标接口 ====================

@app.post(
    "/api/bids/",
    tags=[Tags.bids],
    summary="提交投标",
    status_code=status.HTTP_201_CREATED,
)
def submit_bid():
    """
    提交一份投标，系统会自动：

    1. 验证报价是否超过控制价
    2. 检查投标截止时间
    3. 计算 AI 推荐的中标概率
    """
    return {"bid_id": "001"}


@app.get(
    "/api/bids/{bid_id}/analysis",
    tags=[Tags.bids, Tags.analysis],
    summary="分析投标竞争力",
    response_description="竞争力分析报告",
)
def analyze_bid(bid_id: str):
    """
    对指定投标进行竞争力分析，返回：

    - **win_probability**：中标概率
    - **risk_level**：风险等级
    - **suggestions**：优化建议列表
    """
    return {"win_probability": 0.75}


# ==================== 弃用的接口 ====================

@app.get(
    "/api/v1/bid-recommend/",
    tags=[Tags.bids],
    deprecated=True,
    summary="[已弃用] 获取投标推荐价",
)
def old_bid_recommend():
    """
    ⚠️ 此接口已弃用，请使用 `POST /api/bids/{bid_id}/analysis`

    将在 2026 年 6 月 30 日后下线。
    """
    return {"recommended_price": 1100}
```

`/docs` 页面效果：

```css
▼ 项目管理
    POST /api/projects/           创建项目
    GET  /api/projects/           查询项目列表

▼ 投标管理
    POST /api/bids/               提交投标
    GET  /api/bids/{bid_id}/analysis   分析投标竞争力
    GET  /api/v1/bid-recommend/   ̶[̶已̶弃̶用̶]̶ ̶获̶取̶投̶标̶推̶荐̶价̶   ← 删除线，灰色

▼ 数据分析
    GET  /api/bids/{bid_id}/analysis   分析投标竞争力（同时出现在两个分组）
```

---

# 第二部分：JSON 兼容编码器

## 6. 问题：Python 对象存不进 JSON

### 6.1 场景

你有一个 Pydantic 模型对象，想存进数据库（假设数据库只接受 dict）或者用 `json.dumps()` 序列化：

```python
from datetime import datetime
from pydantic import BaseModel


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: str | None = None


item = Item(title="报告", timestamp=datetime.now())

# ❌ 直接存？数据库不认识 Pydantic 模型对象
fake_db["item1"] = item   # 存进去的是 Item 对象，不是 dict

# ❌ 直接 json.dumps？datetime 不能序列化
import json
json.dumps(item)   # TypeError: Object of type Item is not JSON serializable
```

### 6.2 `model_dump()` 不够用

```python
item_dict = item.model_dump()
print(item_dict)
# {"title": "报告", "timestamp": datetime(2025, 6, 15, 10, 30), "description": None}
#                                  ↑ 还是 datetime 对象！

json.dumps(item_dict)   # ❌ TypeError: datetime 不能序列化
```

`model_dump()` 把 Pydantic 模型变成了字典，但字典里的 `datetime` 值还是 Python 对象，依然不能直接用 `json.dumps()`

---

## 7. 解决方案：`jsonable_encoder()`

### 7.1 基本用法

```python
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from pydantic import BaseModel


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: str | None = None


item = Item(title="报告", timestamp=datetime(2025, 6, 15, 10, 30))

# jsonable_encoder 一步到位
json_data = jsonable_encoder(item)
print(json_data)
# {
#     "title": "报告",
#     "timestamp": "2025-06-15T10:30:00",    ← datetime 变成了 ISO 8601 字符串
#     "description": None
# }

# 现在可以安全地 json.dumps 了
import json
json.dumps(json_data)   # ✅ 没问题
```

### 7.2 `jsonable_encoder` 做了什么

```css
输入（Python 对象）              输出（JSON 兼容的值）
─────────────────────────────────────────────────────
Pydantic 模型                   → dict
datetime                        → ISO 8601 字符串
date                            → ISO 8601 字符串
UUID                            → 字符串
Decimal                         → float
set                             → list
bytes                           → base64 字符串
Enum                            → 枚举的值
嵌套的 Pydantic 模型             → 嵌套的 dict
```

**一句话**：把所有 Python 特有的类型都转成 JSON 能表示的基本类型（str、int、float、bool、list、dict、None）

### 7.3 和 `model_dump()` 的区别

```python
item = Item(title="报告", timestamp=datetime(2025, 6, 15, 10, 30))

# model_dump()：Pydantic 模型 → dict，但值可能还是 Python 对象
item.model_dump()
# {"title": "报告", "timestamp": datetime(2025, 6, 15, 10, 30)}
#                                ↑ 还是 datetime 对象

# jsonable_encoder()：任何对象 → 完全 JSON 兼容的结构
jsonable_encoder(item)
# {"title": "报告", "timestamp": "2025-06-15T10:30:00"}
#                                ↑ 已经是字符串
```

```css
                model_dump()              jsonable_encoder()
──────────────────────────────────────────────────────────────
输入            只能是 Pydantic 模型       任何 Python 对象
输出            dict（值可能是 Python 对象）  dict（值全是 JSON 基本类型）
datetime → ?    还是 datetime 对象          ISO 8601 字符串
UUID → ?        还是 UUID 对象              字符串
能直接 json.dumps？  不一定                  一定可以
```

**简单规则**：

- 只是想把 Pydantic 模型变成字典在 Python 里用 → `model_dump()`
- 需要存进只接受 JSON 的地方（数据库、缓存、外部 API） → `jsonable_encoder()`

### 7.4 实际用法：存进数据库

```python
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()
fake_db = {}   # 模拟一个只接受 dict 的数据库


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: str | None = None


@app.put("/items/{id}")
def update_item(id: str, item: Item):
    # 把 Pydantic 模型转成 JSON 兼容的 dict
    json_data = jsonable_encoder(item)
    # 存进"数据库"
    fake_db[id] = json_data
    return json_data
```

### 7.5 `jsonable_encoder` 也支持参数过滤

和 `model_dump()` 类似，它也支持 `include`、`exclude`、`exclude_unset` 等参数：

```python
from fastapi.encoders import jsonable_encoder

json_data = jsonable_encoder(
    item,
    exclude={"description"},          # 排除某些字段
    exclude_unset=True,               # 排除没设过值的字段
)
```

---

## 8. 对应到你的投标项目

```python
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

app = FastAPI()

# 模拟数据库
fake_db: dict[str, dict] = {}


class BidRecord(BaseModel):
    """投标记录"""
    id: UUID
    project_name: str
    bid_price: Decimal
    created_at: datetime
    strategy: str = "moderate"
    won: bool | None = None


@app.post("/api/bids/")
def create_bid(project_name: str, bid_price: float):
    """
    创建投标记录并存入数据库
    """
    bid = BidRecord(
        id=uuid4(),
        project_name=project_name,
        bid_price=Decimal(str(bid_price)),
        created_at=datetime.now(),
    )

    # 要存进数据库 → 用 jsonable_encoder 转换
    bid_data = jsonable_encoder(bid)
    # bid_data 现在是：
    # {
    #     "id": "550e8400-e29b-...",          ← UUID → 字符串
    #     "project_name": "南昌路面改造",
    #     "bid_price": 1135.5,                ← Decimal → float
    #     "created_at": "2025-06-15T10:30:00", ← datetime → 字符串
    #     "strategy": "moderate",
    #     "won": null
    # }

    fake_db[bid_data["id"]] = bid_data
    return bid_data


@app.put("/api/bids/{bid_id}")
def update_bid(bid_id: str, project_name: str | None = None, won: bool | None = None):
    """
    部分更新投标记录——只更新传了值的字段
    """
    if bid_id not in fake_db:
        raise HTTPException(status_code=404, detail="投标记录不存在")

    stored_data = fake_db[bid_id]

    # 只更新传了值的字段
    if project_name is not None:
        stored_data["project_name"] = project_name
    if won is not None:
        stored_data["won"] = won

    fake_db[bid_id] = stored_data
    return stored_data
```

---

## 9. 本节核心知识点速查

### 路径操作配置

```css
参数                    作用                       示例
──────────────────────────────────────────────────────────────────
tags=["标签"]            接口分组                   tags=["项目管理"]
summary="摘要"          一句话概括                  summary="创建项目"
description="描述"      详细说明                   写在 docstring 里更方便
response_description    描述响应内容               response_description="创建成功的项目"
deprecated=True         标记为已弃用               文档上画删除线，功能不受影响
status_code=201         默认状态码                 第 17 节学过
response_model=Item     响应模型                   第 15 节学过
```

### JSON 兼容编码器

```css
需求                              做法
──────────────────────────────────────────────────────────────────
Pydantic 模型 → dict（值可能不兼容 JSON）   model_dump()
任何 Python 对象 → JSON 兼容的结构           jsonable_encoder()
存进只接受 dict/JSON 的数据库               jsonable_encoder(item)
发给只接受 JSON 的外部 API                  jsonable_encoder(data)
```

**本节最核心的三句话**：

1. 用 `tags` 给接口分组、用 docstring 写描述（支持 Markdown）、用 `deprecated=True` 标记弃用——这些元数据让你的 `/docs` 页面从"能用"变成"好用"

2. `jsonable_encoder()` 把任何 Python 对象转成 JSON 兼容的结构——Pydantic 模型变 dict，datetime 变字符串，UUID 变字符串，Decimal 变 float。需要把数据存进数据库或发给外部 API 时用它

3. `model_dump()` 和 `jsonable_encoder()` 的区别：`model_dump()` 只转第一层（模型 → dict），值可能还是 Python 对象；`jsonable_encoder()` 递归转到底，保证每个值都是 JSON 基本类型