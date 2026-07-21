# LangChain 工具（Tools）深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第四节。第一节学了 Agent（整体架构），第二节学了 Model（大脑），第三节学了 Messages（消息/纸条），本节学习 Tools——Agent 的"手和脚"。如果说模型是大脑负责思考，那工具就是让 Agent 真正能"做事"的能力

---

## 第一章：重新理解工具——从"会说话"到"会做事"

### 1.1 没有工具的模型 vs 有工具的模型

在前几节中，我们已经多次提到工具。现在让我们从工具自身的角度，彻底搞清楚它的全部细节

**没有工具的模型：** 只能基于训练数据生成文本。你问它"现在几点"，它只能说"我无法获取实时时间"

**有工具的模型：** 可以调用外部函数来获取信息或执行操作。你问它"现在几点"，它会调用一个获取时间的函数，拿到结果后告诉你准确时间

```css
没有工具：
  用户 → "现在几点？" → 模型 → "抱歉，我无法获取实时时间。"

有工具：
  用户 → "现在几点？" → 模型 → 调用 get_time() → "14:30"
                         → 模型 → "现在是下午 2 点 30 分。"
```

### 1.2 工具的本质

工具在底层其实就是一个**带有完善描述信息的 Python 函数**。它需要告诉模型三件事：

1. **我叫什么名字**——让模型知道如何引用我
2. **我能做什么**——让模型知道什么时候该用我
3. **我需要什么参数**——让模型知道调用我时该传什么

```python
@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query."""
    #    ↑ 函数名 = 工具名
    #                          ↑ 参数名和类型 = 输入 schema
    #    ↑ docstring = 工具描述
    return f"Found {limit} results for '{query}'"
```

模型看到的是一个"工具说明卡片"：

```
┌────────────────────────────────────────────┐
│ 工具名: search_database                     │
│ 描述: Search the customer database for      │
│       records matching the query.           │
│ 参数:                                       │
│   - query (string, 必填): 搜索关键词         │
│   - limit (integer, 可选, 默认10): 最大结果数 │
└────────────────────────────────────────────┘
```

模型根据这张"说明卡片"来决定何时使用这个工具、传什么参数

---

## 第二章：创建工具——从基础到高级

### 2.1 基础工具定义：`@tool` 装饰器

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

**`@tool` 装饰器**

这是 LangChain 提供的核心装饰器。它的作用是把一个普通的 Python 函数"升级"成 LangChain 的标准化工具

```css
普通函数 search_database
       │
       │ @tool 装饰器处理
       ▼
标准化工具对象，包含：
  ├── .name = "search_database"           ← 从函数名提取
  ├── .description = "Search the..."      ← 从 docstring 提取
  ├── .args_schema = {query: str, ...}    ← 从类型注解提取
  └── .__call__ = 原函数的执行逻辑         ← 保留原始功能
```

**类型注解是必填的**

```python
# ✓ 正确——有类型注解
def search_database(query: str, limit: int = 10) -> str:

# ✗ 错误——没有类型注解，LangChain 无法生成输入 schema
def search_database(query, limit=10):
```

模型需要知道每个参数的类型才能正确地构造调用参数。没有类型注解，LangChain 就不知道 `query` 是字符串还是数字

**docstring 的重要性**

docstring（文档字符串）会成为工具的描述，直接影响模型何时选择使用这个工具

```python
# 差的 docstring——模型不知道这个工具干什么
@tool
def func(query: str) -> str:
    """A function."""
    ...

# 好的 docstring——模型清楚知道何时该用它
@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    ...
```

**Args 部分的作用：** docstring 中的 `Args:` 段落为每个参数提供额外描述。模型读到这些描述，更准确地理解每个参数该填什么

**工具命名规范**

```python
# ✓ 推荐：snake_case（下划线分隔）
@tool
def web_search(query: str) -> str: ...

# ✗ 避免：空格或特殊字符（部分提供商会报错）
@tool("Web Search")
def search(query: str) -> str: ...
```

### 2.2 自定义工具名称和描述

默认情况下，工具名来自函数名，描述来自 docstring。但你可以覆盖它们：

```python
# 自定义工具名
@tool("web_search")  # 工具名变成 "web_search"，而不是函数名 "search"
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # "web_search"
```

**为什么要自定义名称？**

有时函数名是缩写或内部命名（比如 `srch_v2`），但你想给模型看一个更清晰的名字（`web_search`）

```python
# 同时自定义名称和描述
@tool("calculator", description="执行算术计算。遇到任何数学问题都用这个工具。")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

==当同时传了 `description` 参数和写了 docstring 时，`description` 参数会覆盖 docstring 作为工具描述。docstring 此时只作为代码文档供开发者阅读==

### 2.3 高级输入 schema：使用 Pydantic 定义复杂参数

当工具的参数比较复杂（有枚举值、嵌套结构、详细的字段描述）时，用 Pydantic 模型来定义输入 schema 更加精确：

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool

class WeatherInput(BaseModel):
    """天气查询的输入参数。"""
    location: str = Field(
        description="城市名称或坐标"
    )
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位偏好"
    )
    include_forecast: bool = Field(
        default=False,
        description="是否包含 5 天预报"
    )

@tool(args_schema=WeatherInput)
def get_weather(
    location: str,
    units: str = "celsius",
    include_forecast: bool = False
) -> str:
    """获取当前天气和可选的预报信息。"""
    temp = 22 if units == "celsius" else 72
    result = f"当前 {location} 天气: {temp} 度 {'摄氏' if units == 'celsius' else '华氏'}"
    if include_forecast:
        result += "\n未来 5 天: 晴天"
    return result
```

**使用 Pydantic schema 的好处：**

1. **`Field(description=...)`** 为每个参数提供详细描述，模型能更准确地填参数
2. **`Literal["celsius", "fahrenheit"]`** 限制参数只能是这两个值之一，模型不会传其他值
3. **`default=False`** 明确指定默认值，参数变成可选的

**模型看到的"说明卡片"变成了：**

```css
┌──────────────────────────────────────────────────┐
│ 工具名: get_weather                               │
│ 描述: 获取当前天气和可选的预报信息。                │
│ 参数:                                             │
│   - location (string, 必填): 城市名称或坐标         │
│   - units (enum: "celsius"|"fahrenheit",          │
│            默认 "celsius"): 温度单位偏好            │
│   - include_forecast (boolean,                    │
│            默认 false): 是否包含 5 天预报           │
└──────────────────────────────────────────────────┘
```

对比没有 Pydantic 时的简陋描述，这张"说明卡片"详细得多，模型的调用准确性也会显著提高

### 2.4 保留参数名——不能用的名字

有两个参数名被 LangChain 内部占用，你不能用它们作为工具参数名：

| 保留名称 | 被谁占用 | 用途 |
|:-------:|:-------:|:----:|
| `config` | LangChain 内部 | 传递 `RunnableConfig`（回调、标签等） |
| `runtime` | `ToolRuntime` | 访问状态、上下文、存储（下面详细讲） |

```python
# ✗ 错误——config 是保留名称
@tool
def my_tool(config: str) -> str:  # 运行时会报错！
    ...

# ✓ 正确——换个名字
@tool
def my_tool(settings: str) -> str:
    ...
```

---

## 第三章：访问上下文——让工具"知道更多"

### 3.1 为什么工具需要上下文？

到目前为止，我们定义的工具都是"孤立的"——它们只知道模型传给它们的参数。但在实际应用中，工具经常需要知道更多信息：

- "当前用户是谁？"（查权限、个性化回答）
- "之前的对话说了什么？"（基于上下文做决策）
- "这个用户上次的偏好是什么？"（跨会话记忆）

**`ToolRuntime` 就是解决这些问题的。** 它是 LangChain 注入给工具的一个"信息包"，包含了工具运行时需要的各种上下文信息

### 3.2 ToolRuntime 全景图

```css
ToolRuntime（工具运行时信息包）
│
├── .state          短期记忆（当前对话的状态）
│   ├── ["messages"]       对话历史
│   └── ["自定义字段"]     你定义的额外状态
│
├── .context        运行时上下文（不可变配置）
│   ├── .user_id          用户 ID
│   └── .其他字段          会话信息等
│
├── .store          长期记忆（跨会话持久存储）
│   ├── .get(...)          读取数据
│   └── .put(...)          保存数据
│
├── .stream_writer  流式写入器（实时进度更新）
│
├── .config         运行配置（回调、标签等）
│
└── .tool_call_id   当前工具调用的唯一 ID
```

**核心理解：** `runtime` 参数**对模型不可见**。模型不知道它的存在，也不会尝试为它传参。LangChain 在执行工具时自动注入它

```python
@tool
def my_tool(query: str, runtime: ToolRuntime) -> str:
    """搜索信息。"""
    # 模型只看到一个参数：query (string)
    # runtime 是 LangChain 自动注入的，模型不知道
    user_id = runtime.context.user_id
    ...
```

### 3.3 短期记忆（State）——"这轮对话发生了什么？"

State 是当前对话的状态，存在于一次对话的生命周期内。最重要的内容是 `messages`（对话历史），还可以包含自定义的字段

#### 读取状态

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """获取用户最近一条消息的内容。"""
    messages = runtime.state["messages"]

    # 从后往前遍历，找到最近的 HumanMessage
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    return "没有找到用户消息"
```

**这个工具的执行流程：**

```
用户: "帮我查一下之前问了什么"
     │
     ▼
模型判断需要调用 get_last_user_message
     │
     ▼
LangChain 执行工具，自动注入 runtime:
  runtime.state["messages"] = [
      SystemMessage("你是一个助手"),
      HumanMessage("你好"),
      AIMessage("你好！"),
      HumanMessage("天气怎么样？"),
      AIMessage("让我查一下..."),
      HumanMessage("帮我查一下之前问了什么"),  ← 当前消息也在里面
  ]
     │
     ▼
工具从后往前遍历，跳过当前消息，找到 "天气怎么样？"
     │
     ▼
返回 "天气怎么样？"
     │
     ▼
模型生成回答: "你之前问的是'天气怎么样？'"
```

#### 读取自定义状态字段

```python
@tool
def get_user_preference(
    pref_name: str,           # 模型传入的参数（可见）
    runtime: ToolRuntime      # LangChain 注入的（不可见）
) -> str:
    """获取用户偏好设置的值。"""
    # 访问自定义状态字段 user_preferences
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "未设置")
```

**`runtime.state.get("user_preferences", {})`** 这行代码安全地获取自定义字段 `user_preferences`。如果这个字段不存在（比如调用者没有传入），就返回空字典 `{}`，避免报错

#### 更新状态：Command 对象

有时候工具不仅需要返回结果，还需要**修改 Agent 的状态**。比如用户说"我叫小明"，你希望工具把这个名字记录到状态中，后续的对话都能用到

```python
from langgraph.types import Command
from langchain.tools import tool

@tool
def set_user_name(new_name: str) -> Command:
    """设置用户在对话中的名字。"""
    # 返回 Command 而不是普通字符串
    # Command 会告诉 Agent 更新状态
    return Command(update={"user_name": new_name})
```

**Command 是什么？**

`Command` 是 LangGraph 提供的一种特殊返回值。当工具返回 `Command` 而不是普通字符串时，LangGraph 会执行 Command 中指定的操作——比如更新状态

**普通返回 vs Command 返回的区别：**

```
普通返回（字符串）:
  工具返回 "操作成功"
  → 变成 ToolMessage(content="操作成功")
  → 模型看到这条消息
  → 状态不变

Command 返回:
  工具返回 Command(update={"user_name": "小明"})
  → Agent 状态中的 user_name 被更新为 "小明"
  → 后续所有工具和中间件都能看到 user_name = "小明"
```

### 3.4 运行时上下文（Context）——"谁在用我？"

Context 提供**不可变的配置信息**，在调用 Agent 时传入，整个对话过程中不会改变。最常见的用途是传入用户身份信息

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 模拟一个用户数据库
USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

# 定义上下文结构
@dataclass
class UserContext:
    user_id: str

# 工具通过 ToolRuntime 访问上下文
@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """获取当前用户的账户信息。"""
    # 从上下文中获取 user_id
    # 注意：模型不需要传 user_id，它来自运行时注入
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return (
            f"账户持有人: {user['name']}\n"
            f"类型: {user['account_type']}\n"
            f"余额: ${user['balance']}"
        )
    return "用户未找到"

# 创建 Agent
model = ChatOpenAI(model="gpt-4.1")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,    # 告诉 Agent 上下文的格式
    system_prompt="你是一个金融助手。"
)

# 调用时传入上下文
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我的余额是多少？"}]},
    context=UserContext(user_id="user123")   # ← 上下文在这里传入
)
```

**完整流程追踪：**

```
用户: "我的余额是多少？"
上下文: UserContext(user_id="user123")
     │
     ▼
模型收到的信息：
  ┌──────────────────────────────────────┐
  │ 可用工具:                             │
  │   get_account_info()                 │
  │   描述: 获取当前用户的账户信息          │
  │   参数: 无（runtime 对模型不可见）     │
  └──────────────────────────────────────┘

模型思考: "用户问余额，我有 get_account_info 工具，且它不需要参数"
     │
     ▼
模型返回: tool_calls = [{name: "get_account_info", args: {}}]
     │
     ▼
LangChain 执行工具:
  自动注入 runtime，其中 runtime.context = UserContext(user_id="user123")
  工具内部用 runtime.context.user_id 获取 "user123"
  在 USER_DATABASE 中查到 Alice 的信息
     │
     ▼
工具返回: "账户持有人: Alice Johnson\n类型: Premium\n余额: $5000"
     │
     ▼
模型生成最终回答: "您的账户余额是 $5,000，账户类型是 Premium。"
```

**关键理解：** `user_id` 不是模型传给工具的——模型根本不知道 `user_id` 的存在。`user_id` 是你的应用程序在调用 Agent 时通过 `context` 参数传入的。这保证了安全性——用户不能通过对话来伪造其他人的身份

**`ToolRuntime[UserContext]` 中的方括号是什么？**

这是 Python 的**泛型语法**。`ToolRuntime[UserContext]` 告诉 Python 和 LangChain："这个 runtime 的 context 属性是 `UserContext` 类型"。这样：
- IDE 能在你输入 `runtime.context.` 时自动补全 `user_id`
- LangChain 知道在注入 runtime 时，context 应该是 `UserContext` 的实例

### 3.5 长期记忆（Store）——"跨对话记住用户"

State 是"短期记忆"——对话结束就没了。Store 是"长期记忆"——数据保存在持久化存储中，下次对话还能访问

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

# 读取长期记忆的工具
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """查询用户信息。"""
    store = runtime.store
    # store.get(命名空间, 键)
    # 命名空间类似于数据库中的"表名"
    # 键类似于数据库中的"主键"
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "未知用户"

# 写入长期记忆的工具
@tool
def save_user_info(
    user_id: str,
    user_info: dict[str, Any],
    runtime: ToolRuntime
) -> str:
    """保存用户信息。"""
    store = runtime.store
    # store.put(命名空间, 键, 值)
    store.put(("users",), user_id, user_info)
    return "用户信息保存成功。"

# 创建 Agent 并传入 store
model = ChatOpenAI(model="gpt-4.1")
store = InMemoryStore()  # 内存存储（测试用）

agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store           # ← 在这里注入 store
)
```

**使用场景演示：**

```python
# 第一次对话：保存用户信息
agent.invoke({
    "messages": [{
        "role": "user",
        "content": "保存用户: ID是abc123, 名字是小明, 年龄25, 邮箱 ming@example.com"
    }]
})
# 模型调用 save_user_info 工具
# 数据被保存到 store 中

# 第二次对话（全新的对话！）：读取用户信息
agent.invoke({
    "messages": [{
        "role": "user",
        "content": "查一下用户 abc123 的信息"
    }]
})
# 模型调用 get_user_info 工具
# 从 store 中读取到之前保存的数据
# → "名字: 小明, 年龄: 25, 邮箱: ming@example.com"
```

**Store 的数据组织方式——命名空间 + 键：**

```
store 的结构类似于一个嵌套字典：

store
├── ("users",)                    ← 命名空间（元组）
│   ├── "abc123" → {name: "小明", age: 25, ...}    ← 键 → 值
│   └── "def456" → {name: "小红", age: 30, ...}
│
├── ("preferences",)              ← 另一个命名空间
│   ├── "abc123" → {language: "zh", theme: "dark"}
│   └── "def456" → {language: "en", theme: "light"}
│
└── ("sessions", "abc123")        ← 命名空间可以有多级
    ├── "session_001" → {start_time: "...", ...}
    └── "session_002" → {start_time: "...", ...}
```

**`InMemoryStore` vs 生产环境的存储：**

| 存储类型 | 特点 | 适用场景 |
|---------|------|---------|
| `InMemoryStore` | 程序重启就丢失 | 开发测试 |
| `PostgresStore` | 持久化到数据库 | 生产环境 |

### 3.6 流式写入器（Stream Writer）——"告诉用户我在忙"

当工具需要执行耗时操作（比如搜索大量数据、调用外部 API）时，用户可能要等很久。Stream Writer 让工具能在执行过程中发送实时进度更新：

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """获取指定城市的天气。"""
    writer = runtime.stream_writer

    # 在工具执行过程中发送进度更新
    writer(f"正在查询 {city} 的数据...")
    # ... 执行耗时操作 ...
    writer(f"已获取 {city} 的天气数据")

    return f"{city} 今天是晴天！"
```

**用户看到的效果：**

```
用户: "查一下北京天气"
  → "正在查询 北京 的数据..."      ← 实时进度
  → "已获取 北京 的天气数据"        ← 实时进度
  → "北京今天是晴天，气温22°C！"   ← 最终回答
```

### 3.7 ToolRuntime 各组件对比总结

| 组件 | 生命周期 | 可变性 | 用途 | 类比 |
|:----:|:-------:|:-----:|:----:|:----:|
| `state` | 当前对话 | 可读可写（通过 Command） | 对话历史、临时数据 | 会议笔记本 |
| `context` | 当前调用 | 只读 | 用户身份、配置 | 工牌/门禁卡 |
| `store` | 永久 | 可读可写 | 用户偏好、知识库 | 档案柜 |
| `stream_writer` | 当前执行 | 只写 | 实时进度更新 | 对讲机 |
| `tool_call_id` | 当前调用 | 只读 | 唯一标识当前调用 | 工单号 |

---

## 第四章：工具的返回值——不仅仅是字符串

### 4.1 三种返回方式

工具可以返回三种类型的值，每种有不同的行为。

#### 返回字符串——最常见

```python
@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 目前是晴天。"
```

**行为：**
- 返回值被包装成 `ToolMessage(content="北京目前是晴天。")`
- 模型读到这段文字，根据它生成回答
- 不会修改 Agent 的状态

**适用场景：** 结果是自然语言文本、给模型读的描述。

#### 返回对象（dict）——结构化数据

```python
@tool
def get_weather_data(city: str) -> dict:
    """获取结构化天气数据。"""
    return {
        "city": city,
        "temperature_c": 22,
        "conditions": "sunny",
    }
```

**行为：**
- 字典被序列化成字符串后放入 ToolMessage
- 模型可以从中提取具体字段来推理
- 同样不会修改 Agent 状态

**适用场景：** 当你需要模型精确地处理特定字段，而不是从自由文本中"猜测"信息。

**字符串 vs 字典的实际差异：**

```css
字符串返回:
  ToolMessage(content="北京目前是晴天，气温22°C。")
  → 模型需要从文字中"理解"温度是22度

字典返回:
  ToolMessage(content='{"city": "北京", "temperature_c": 22, "conditions": "sunny"}')
  → 模型直接看到 temperature_c = 22，不需要"理解"
```

#### 返回 Command——修改状态

```python
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """设置用户偏好的响应语言。"""
    return Command(
        update={
            # 更新状态中的 preferred_language 字段
            "preferred_language": language,
            # 同时返回一条 ToolMessage 让模型知道操作结果
            "messages": [
                ToolMessage(
                    content=f"语言已设置为 {language}。",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

**行为：**
- `Command(update={...})` 中的字段会被写入 Agent 状态
- 如果 `update` 中包含 `messages`，模型会看到这些消息
- 状态更新对后续步骤立即生效

**`runtime.tool_call_id` 是什么？**

每次模型调用工具时，都会生成一个唯一的调用 ID（如 `call_abc123`）。当你在 Command 中返回 ToolMessage 时，需要把这个 ID 填入 `tool_call_id`，让模型能把结果和对应的调用匹配起来。

`runtime.tool_call_id` 就是当前这次工具调用的 ID。

**完整的流程：**

```
用户: "把语言切换成英文"
     │
     ▼
模型调用 set_language(language="en")
  tool_call_id = "call_xyz789"
     │
     ▼
工具返回 Command(update={
    "preferred_language": "en",      ← 更新状态
    "messages": [ToolMessage(
        content="语言已设置为 en。",
        tool_call_id="call_xyz789"   ← 匹配调用 ID
    )]
})
     │
     ▼
Agent 处理 Command:
  1. 把 preferred_language 更新为 "en"
  2. 把 ToolMessage 加入消息列表
     │
     ▼
模型看到 ToolMessage，生成回答:
  "好的，我已经将语言切换为英文了。"
```

---

## 第五章：ToolNode——在自定义工作流中使用工具

### 5.1 `create_agent` vs `ToolNode`

到目前为止，我们一直通过 `create_agent` 来使用工具——它帮你封装了整个 ReAct 循环。但如果你需要构建更复杂的自定义工作流，就需要用 `ToolNode`。

```
create_agent:
  "我帮你搞定一切——模型调用、工具执行、循环控制"
  → 适合快速上手、标准 Agent 场景

ToolNode:
  "我只负责执行工具，其他的你自己安排"
  → 适合自定义工作流、需要精细控制的场景
```

### 5.2 基本用法

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"关于 '{query}' 的搜索结果"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    return str(eval(expression))

# 创建 ToolNode——它是一个"节点"，可以放入 LangGraph 的图中
tool_node = ToolNode([search, calculator])

# 在 LangGraph 的工作流中使用
builder = StateGraph(MessagesState)
builder.add_node("tools", tool_node)  # 添加工具节点
# ... 添加其他节点和边 ...
```

**ToolNode 自动处理的事情：**
- **并行执行：** 如果模型同时请求了多个工具调用，ToolNode 自动并行执行它们
- **状态注入：** 自动把 `ToolRuntime` 注入到需要它的工具中
- **错误处理：** 可配置的错误处理策略

### 5.3 错误处理配置

```python
from langgraph.prebuilt import ToolNode

# 默认行为：捕获调用错误，但执行错误会抛出
tool_node = ToolNode(tools)

# 捕获所有错误，把错误信息返回给模型
# 模型可以根据错误信息决定下一步（比如换个参数重试）
tool_node = ToolNode(tools, handle_tool_errors=True)

# 自定义错误消息
tool_node = ToolNode(
    tools,
    handle_tool_errors="出了点问题，请换个方式重试。"
)

# 自定义错误处理函数
def handle_error(e: ValueError) -> str:
    return f"输入无效: {e}"

tool_node = ToolNode(tools, handle_tool_errors=handle_error)

# 只捕获特定类型的异常
tool_node = ToolNode(
    tools,
    handle_tool_errors=(ValueError, TypeError)
)
```

**不同配置的效果对比：**

```
handle_tool_errors=False（默认的一部分行为）:
  工具出错 → 异常直接抛出 → Agent 可能崩溃

handle_tool_errors=True:
  工具出错 → 错误信息变成 ToolMessage → 模型收到错误 → 模型可以决定重试或放弃

handle_tool_errors="自定义消息":
  工具出错 → 自定义消息变成 ToolMessage → 模型看到统一的错误提示

handle_tool_errors=handle_error:
  工具出错 → 你的函数处理异常 → 返回值变成 ToolMessage
```

### 5.4 条件路由：`tools_condition`

在自定义工作流中，你经常需要根据"模型是否调用了工具"来决定下一步走哪条路。`tools_condition` 帮你做这个判断：

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)          # 模型节点
builder.add_node("tools", ToolNode(tools)) # 工具节点

builder.add_edge(START, "llm")             # 开始 → 模型
builder.add_conditional_edges(
    "llm",
    tools_condition    # 自动判断：有工具调用 → "tools"，没有 → END
)
builder.add_edge("tools", "llm")           # 工具 → 回到模型

graph = builder.compile()
```

**`tools_condition` 的判断逻辑：**

```
模型的返回值（AIMessage）
     │
     ├── 有 tool_calls → 路由到 "tools" 节点
     │     （模型请求了工具调用，需要执行工具）
     │
     └── 没有 tool_calls → 路由到 END
           （模型直接给出了最终回答）
```

**这个工作流形成了一个循环：**

```
START → llm → (有工具调用?) → tools → llm → (有工具调用?) → ... → END
                   ↑                              │
                   │  工具结果传回模型              │
                   └──────────────────────────────┘
```

这就是 `create_agent` 内部的核心结构。当你用 `create_agent` 时，它自动帮你构建了这个图。当你用 `ToolNode` + `StateGraph` 时，你手动构建这个图，但获得了更多的灵活性。

---

## 第六章：概念串联——工具在 Agent 生态中的位置

### 6.1 知识关系图

```
第一节 Agent ──────────────────────────────
  create_agent() 把所有组件组装在一起
  ReAct 循环管理工具调用
       │
       ├── 第二节 Model ──────────────────
       │     模型是"大脑"，决定何时调用工具
       │     bind_tools() 让模型"看到"工具
       │
       ├── 第三节 Messages ───────────────
       │     工具调用请求 → AIMessage.tool_calls
       │     工具执行结果 → ToolMessage
       │
       └── 第四节 Tools ← 你在这里 ───────
             工具是"手和脚"，执行实际操作
             @tool 定义工具
             ToolRuntime 访问上下文
             Command 更新状态
             ToolNode 在自定义工作流中使用
```

### 6.2 工具定义 → 工具使用的完整链条

```css
① 你定义工具
   @tool
   def get_weather(city: str) -> str:
       """获取城市天气。"""
       return f"{city} 是晴天"

② 注册到 Agent
   agent = create_agent(model, tools=[get_weather])

③ LangChain 从工具中提取描述信息，传给模型
   模型看到: "有一个叫 get_weather 的工具，接受 city 参数"

④ 用户提问
   "北京天气怎么样？"

⑤ 模型决策
   "用户问天气 → 我有天气工具 → 调用它"

⑥ 模型返回 AIMessage
   tool_calls: [{name: "get_weather", args: {city: "北京"}, id: "call_xxx"}]

⑦ Agent 执行工具
   LangChain 注入 ToolRuntime → get_weather(city="北京") → "北京是晴天"

⑧ 结果变成 ToolMessage
   ToolMessage(content="北京是晴天", tool_call_id="call_xxx")

⑨ 模型根据结果生成最终回答
   AIMessage("北京目前是晴天，很适合出门！")
```

---

## 第七章：速查手册

### 工具定义方式

| 方式 | 代码 | 适用场景 |
|------|------|---------|
| 基础装饰器 | `@tool` | 简单工具，参数不多 |
| 自定义名称 | `@tool("my_name")` | 需要与函数名不同的工具名 |
| 自定义描述 | `@tool(description="...")` | 需要覆盖 docstring 的描述 |
| Pydantic schema | `@tool(args_schema=MyModel)` | 复杂参数、需要枚举值和详细描述 |

### ToolRuntime 属性速查

| 属性 | 访问方式 | 类型 | 用途 |
|------|---------|------|------|
| 对话状态 | `runtime.state` | dict | 读取消息历史和自定义状态 |
| 运行时上下文 | `runtime.context` | 你定义的类 | 读取用户 ID 等不可变配置 |
| 持久化存储 | `runtime.store` | BaseStore | 跨会话读写数据 |
| 流式写入器 | `runtime.stream_writer` | callable | 发送实时进度更新 |
| 运行配置 | `runtime.config` | RunnableConfig | 访问回调、标签等 |
| 调用 ID | `runtime.tool_call_id` | str | 当前工具调用的唯一标识 |

### 工具返回值对比

| 返回类型 | 模型是否看到 | 是否更新状态 | 适用场景 |
|---------|------------|------------|---------|
| `str` | 是（变成 ToolMessage） | 否 | 简单文本结果 |
| `dict` | 是（序列化后） | 否 | 结构化数据 |
| `Command` | 取决于是否包含 messages | 是 | 需要修改 Agent 状态 |

### ToolNode 错误处理选项

| 配置 | 效果 |
|------|------|
| 不设置 | 调用错误捕获，执行错误抛出 |
| `handle_tool_errors=True` | 所有错误变成 ToolMessage |
| `handle_tool_errors="消息"` | 错误时返回自定义消息 |
| `handle_tool_errors=func` | 用自定义函数处理错误 |
| `handle_tool_errors=(Error1, Error2)` | 只捕获特定异常类型 |
