# 01 Agent核心概念

## 01.1 Agent到底是什么

Agent = **语言模型** + **工具** + **循环决策逻辑**

一句话总结：Agent 是一个能**自己决定使用哪些工具、以什么顺序使用**的 AI 系统。普通的大模型调用是"一问一答"：你发一条消息，模型回一条消息，结束。Agent 则是一个**循环**：模型可以在回答之前，先调用工具获取信息，拿到结果后再思考，可能再调用另一个工具，如此循环，直到它认为信息足够了，才给出最终回答

## 01.2 ReAct循环

Agent 遵循 **ReAct** 模式。ReAct = **Re**asoning（推理）+ **Act**ing（行动）。这个循环的步骤是：

```css
输入（用户的问题）
    ↓
模型推理：分析问题，决定下一步做什么
    ↓
├── 如果需要更多信息 → 选择并调用一个工具（Action）
│       ↓
│   工具返回结果（Observation，即"观察"）
│       ↓
│   回到模型推理（带着新信息再思考）
│
└── 如果信息足够 → 生成最终回答（Finish）
    ↓
输出（最终答案）
```

文档给出了一个具体的例子，完整拆解：**用户问题：** "找到目前最流行的无线耳机，并检查它们是否有货。"

**第一轮 ReAct：**

- **Reasoning（推理）：** "流行度是时效性信息，我自己不知道，需要用搜索工具。"
- **Acting（行动）：** 调用 `search_products("wireless headphones")`
- **Observation（观察）：** 工具返回 "找到 5 个产品，排名第一是 WH-1000XM5..."

**第二轮 ReAct：**

- **Reasoning（推理）：** "找到了最流行的型号，但用户还问了是否有货，我需要再查一下库存。"
- **Acting（行动）：** 调用 `check_inventory("WH-1000XM5")`
- **Observation（观察）：** 工具返回 "WH-1000XM5: 库存 10 件"

**第三轮 ReAct：**

- **Reasoning（推理）：** "我现在有了产品名和库存信息，可以回答用户了。"
- **Acting（行动）：** 生成最终回答（Finish）

**关键理解：** Agent 不是"一次性"决定调用哪些工具的。它是**逐步决策**的——每次只决定下一步做什么，然后根据新信息再做下一个决定。这就是为什么它比简单的模型调用更强大

## 01.3 `create_agent` 函数——Agent 工厂

```python
from langchain.agents import create_agent

agent = create_agent(
    model=...,            # 语言模型（必填）
    tools=...,            # 工具列表（必填，可以为空列表）
    system_prompt=...,    # 系统提示词（可选）
    name=...,             # Agent 名称（可选）
    middleware=...,       # 中间件列表（可选）
    context_schema=...,   # 上下文格式定义（可选）
    response_format=...,  # 结构化输出格式（可选）
    checkpointer=...,     # 记忆保存器（可选）
    state_schema=...,     # 自定义状态格式（可选）
    store=...,            # 持久化存储（可选）
)
```

`create_agent` 是 LangChain 提供的**核心工厂函数**。"工厂"的意思是：传入各种"零件"（模型、工具、提示词等），它帮你组装成一个完整的、可运行的 Agent

**底层原理：** `create_agent` 内部使用了 **LangGraph** 框架。LangGraph 把 Agent 构建为一个**图（Graph）**，图由**节点（Node）**和**边（Edge）**组成：

- **节点**是处理步骤，比如"调用模型"节点、"执行工具"节点
- **边**是步骤之间的连接，定义了数据如何流转

不需要直接操作 LangGraph，`create_agent` 封装好了。但了解这一点有助于理解为什么 Agent 的 API 和 LangGraph 的 Graph API 是通用的（比如 `invoke()`、`stream()` 等方法）

## 01.4 Agent 的停止条件

Agent 的循环不会无限运行。它会在以下情况停止：

1. **模型发出最终输出：** 模型认为信息足够，不再调用工具，直接生成回答

2. **达到迭代上限：** 预设的最大循环次数（防止无限循环）

---

# 02 核心组件一：Model

模型是 Agent 的"大脑"，负责推理和决策。LangChain 支持两种模型配置方式：**静态模型**和**动态模型**

## 02.1 静态模型（Static Model）——最常用

静态模型就是"创建时定好，运行时不变"的模型

### 1 方式一：用字符串快速指定

```python
from langchain.agents import create_agent

agent = create_agent("openai:gpt-5", tools=tools)
```

**`"openai:gpt-5"` 是什么？**

这是一个**模型标识符字符串**，格式是 `提供商:模型名`。LangChain 会根据这个字符串自动：

1. 识别提供商是 OpenAI
2. 使用你设置的 `OPENAI_API_KEY` 环境变量
3. 创建与 OpenAI 的连接
4. 选择 gpt-5 模型

**自动推断：** 你甚至可以省略提供商前缀：

```python
agent = create_agent("gpt-5", tools=tools)       # 自动推断为 openai:gpt-5
agent = create_agent("claude-sonnet-4-6", tools=tools)  # 自动推断为 anthropic:claude-sonnet-4-6
```

LangChain 内部维护了一个映射表，能根据模型名自动识别提供商。==这里在使用时最好要查一下支持与否自己用的大模型==

### 2 方式二：用模型实例精确配置

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)
agent = create_agent(model, tools=tools)
```

从 `langchain_openai` 包中导入 `ChatOpenAI` 类。每个模型提供商都有自己的包：

- `langchain_openai` → OpenAI 的模型
- `langchain_anthropic` → Anthropic 的 Claude 模型
- `langchain_google_genai` → Google 的 Gemini 模型

需要先安装对应的包（如 `pip install langchain-openai`）

**`ChatOpenAI(...)` 的参数：**`model="gpt-5"`，模型名称。这里只需要模型名，不需要提供商前缀。`temperature=0.1`，温度参数（0.0 ~ 2.0）

|   值    |        效果        |           适用场景           |
| :-----: | :----------------: | :--------------------------: |
|   0.0   | 输出最确定、最一致 | 数据提取、代码生成、事实问答 |
| 0.1-0.3 |     轻微随机性     | 客服回复、生产环境的通用回答 |
| 0.5-0.7 | 平衡确定性和创造性 |      写作辅助、头脑风暴      |
|  1.0+   |      高度随机      |      创意写作、诗歌生成      |

`max_tokens=1000`——模型回答的最大长度（以 token 为单位）。1 token ≈ 4 个英文字符 ≈ 0.75 个英文单词 ≈ 0.5 个中文字。设置上限可以控制成本（API 按 token 计费）和响应时间

`timeout=30`——超时时间，单位秒。如果模型 30 秒内没有响应，就抛出超时错误。在生产环境中很重要，可以防止用户长时间等待

**方式一 vs 方式二的选择：**

|          | 字符串方式     | 实例方式           |
| -------- | -------------- | ------------------ |
| 简洁性   | 一行代码搞定   | 需要额外导入和配置 |
| 可配置性 | 只能指定模型名 | 可以设置所有参数   |
| 适用场景 | 快速原型、测试 | 生产环境           |

## 02.2 动态模型（Dynamic Model）——进阶用法

动态模型允许你在**运行时**根据条件选择不同的模型。比如：简单对话用便宜的小模型，复杂对话用昂贵的大模型

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# 定义两个模型
basic_model = ChatOpenAI(model="gpt-4.1-mini")    # 便宜、快速
advanced_model = ChatOpenAI(model="gpt-4.1")       # 贵、强大

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

**两个模型实例：**

```python
basic_model = ChatOpenAI(model="gpt-4.1-mini")
advanced_model = ChatOpenAI(model="gpt-4.1")
```

提前创建两个模型对象。`gpt-4.1-mini` 更轻量（价格低、速度快），`gpt-4.1` 更强大

**`@wrap_model_call` 装饰器：**

```python
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
```

`@wrap_model_call` 是 LangChain 提供的**中间件装饰器**。它把你的函数变成一个"拦截器"——每次 Agent 要调用模型时，都会先经过你的函数

你的函数接收两个参数：

- `request: ModelRequest`——包含了即将发送给模型的所有信息（消息、工具列表、状态等）
- `handler`——一个函数，调用它才会真正把请求发给模型。你可以在调用前修改请求

返回值是 `ModelResponse`——模型的响应

**决策逻辑：**

```python
message_count = len(request.state["messages"])

if message_count > 10:
    model = advanced_model
else:
    model = basic_model
```

`request.state["messages"]` 获取当前对话的所有消息。如果消息超过 10 条（说明对话变复杂了），切换到高级模型

**`request.override(model=model)`：**

```python
return handler(request.override(model=model))
```

`request.override(model=model)` 创建一个新的请求对象，把模型替换成你选择的模型，其他信息不变。然后调用 `handler(...)` 把修改后的请求发出去

**注册中间件：**

```python
agent = create_agent(
    model=basic_model,        # 默认模型（当中间件不修改时使用）
    tools=tools,
    middleware=[dynamic_model_selection]   # 注册中间件
)
```

`middleware` 参数接收一个列表，你可以注册多个中间件，它们会按顺序执行



**`@wrap_model_call` 到底做了什么？**

它是一个"拦截器"。正常情况下，Agent 调用模型的流程是：

```css
Agent → 直接调用模型 → 得到回答
```

加了 `@wrap_model_call` 之后，流程变成了：

```css
Agent → 你的中间件函数 → 你决定怎么调用模型 → 得到回答
```

函数"包裹"了模型调用。可以在调用前修改请求，在调用后修改响应，或像这个例子一样，换一个完全不同的模型



**`handler` 是什么？**

`handler` 是 LangChain 的一个函数，代表"继续执行原本的流程"。如果调用handler(request)，等于说"按照原来方式处理这个请求"。若调用 handler(request.override(model=xxx))，就等于说"用修改后的请求继续处理"



**必须调用 `handler`。** 如果你不调用它，模型就永远不会被调用，Agent 就卡住了



**`request.override(model=model)` 是什么意思？**

`override` 的意思是"覆盖"。request.override(model=model)`不会修改原来的 `request，而是创建一个新的副本，其中 model` 字段被替换。为什么不直接修改 `request.model = model`？因为在编程中，直接修改传入的参数是一种不好的实践——可能导致意想不到的副作用。创建新副本更安全

---

# 03 核心组件二：Tools

工具让 Agent 能够**采取行动**——不仅仅是生成文本，还能调用 API、查询数据库、执行计算等

## 03.1 Agent 使用工具的能力超越简单的模型调用

Agent 的工具使用能力比"模型绑定工具"更强大，因为 Agent 支持：

| 能力         | 说明                           | 例子                           |
| ------------ | ------------------------------ | ------------------------------ |
| 多次序列调用 | 一个提示触发多个工具调用       | 先查位置，再查天气             |
| 并行调用     | 同时调用多个不相关的工具       | 同时查北京和上海的天气         |
| 动态选择     | 根据前一个工具的结果决定下一步 | 搜索结果不够好，换个关键词再搜 |
| 错误重试     | 工具调用失败时自动重试         | API 超时后重新调用             |
| 状态持久化   | 跨工具调用保持状态             | 记住之前搜索的结果             |

## 03.2 静态工具（Static Tools）——最常用

静态工具就是"创建 Agent 时定义好，运行时不变"的工具

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

**`@tool` 装饰器详解：**

`@tool` 装饰器是 LangChain 提供的工具定义方式。它做了以下事情：

1. **提取函数名** → 作为工具名（`search`、`get_weather`）
2. **提取 docstring** → 作为工具描述（模型据此判断何时使用）
3. **提取参数类型** → 作为工具的输入格式（模型据此知道要传什么参数）
4. **包装成标准工具对象** → LangChain 内部统一管理

**关于工具的命名规范：**

文档特别提醒：工具名应该使用 `snake_case`（下划线命名法），比如 `search_products` 而不是 `Search Products`。因为某些模型提供商会拒绝包含空格或特殊字符的工具名

**工具还可以是普通函数或协程：**

```python
# 普通函数（同步）
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

# 协程（异步）——用于需要异步操作的场景
async def search(query: str) -> str:
    """Search for information."""
    result = await some_async_api_call(query)
    return result
```

**空工具列表：**

```python
agent = create_agent(model, tools=[])
```

如果传入空列表，Agent 就变成了一个纯粹的对话模型，没有工具调用能力——相当于只有一个 LLM 节点。

## 03.3 动态工具（Dynamic Tools）——进阶用法

动态工具是指在运行时才决定哪些工具可用。这在以下场景中很有用：

- **权限控制：** 管理员可以用删除工具，普通用户不行
- **功能开关：** 新功能上线前，只对部分用户开放
- **对话阶段：** 在用户验证身份前，只提供公开工具

### 1 方式一：过滤预注册的工具

所有可能的工具在创建时就注册了，运行时根据条件决定暴露哪些给模型。

**场景 A：根据对话状态过滤**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    if not is_authenticated:
        # 未认证 → 只提供公开工具
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # 对话前期 → 限制高级工具
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4.1",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)
```

**逐行详解：**

`request.state`——获取 Agent 当前的状态。状态是一个字典，包含 `messages`（消息列表）和你自定义的其他字段

`state.get("authenticated", False)`——安全地获取 `authenticated` 字段，如果不存在就默认为 `False`

`[t for t in request.tools if t.name.startswith("public_")]`——这是 Python 的列表推导式。它从所有注册的工具中，筛选出名字以 `"public_"` 开头的工具。比如：

- `public_search` → 保留
- `private_search` → 过滤掉
- `advanced_search` → 过滤掉

`request.override(tools=tools)`——创建一个新的请求，把工具列表替换成过滤后的列表。这样模型就只能"看到"被允许的工具

**场景 B：根据运行时上下文（权限）过滤**

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_role: str    # "admin" / "editor" / "viewer"

@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Runtime Context permissions."""
    if request.runtime is None or request.runtime.context is None:
        user_role = "viewer"    # 没有上下文 → 最严格的权限
    else:
        user_role = request.runtime.context.user_role

    if user_role == "admin":
        pass    # 管理员拥有所有工具，不做任何过滤
    elif user_role == "editor":
        # 编辑者不能删除
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # 查看者只能用只读工具
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4.1",
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=Context
)
```

**这里的关键概念：**

`request.runtime`——运行时对象，包含你在调用 `agent.invoke()` 时传入的 `context`

`request.runtime.context`——你定义的 `Context` 对象。在这个例子中，它包含 `user_role` 字段

这样就可以根据调用时传入的用户角色，动态决定 Agent 有哪些工具可用：

```python
# 管理员调用 → 拥有所有工具
agent.invoke({"messages": [...]}, context=Context(user_role="admin"))

# 普通查看者调用 → 只有 read_data 工具
agent.invoke({"messages": [...]}, context=Context(user_role="viewer"))
```

### 2 方式二：运行时注册新工具

当工具**不是提前知道的**，而是在运行时才发现的（比如从远程服务器加载），就需要这种方式。

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest

# 一个将在运行时动态添加的工具
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # 在模型调用前，把动态工具添加到工具列表中
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # 当模型调用动态工具时，告诉 Agent 如何执行它
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],              # 只注册静态工具
    middleware=[DynamicToolMiddleware()],   # 动态工具由中间件管理
)
```

**这段代码的核心概念：**

**`AgentMiddleware` 类：**之前用的是 `@wrap_model_call` 装饰器来创建单个中间件函数。但当需要同时拦截多个环节时（比如既要修改模型请求，又要处理工具调用），就需要用类的方式继承 `AgentMiddleware`

**`wrap_model_call` 方法：**

```python
def wrap_model_call(self, request: ModelRequest, handler):
    updated = request.override(tools=[*request.tools, calculate_tip])
    return handler(updated)
```

每次模型被调用前，这个方法会执行。它把 `calculate_tip` 添加到工具列表中，这样模型就能"看到"这个动态工具

`[*request.tools, calculate_tip]` 是 Python 的解包语法。假设 `request.tools` 是 `[get_weather]`，那么结果就是 `[get_weather, calculate_tip]`

**`wrap_tool_call` 方法：**

```python
def wrap_tool_call(self, request: ToolCallRequest, handler):
    if request.tool_call["name"] == "calculate_tip":
        return handler(request.override(tool=calculate_tip))
    return handler(request)
```

当模型决定调用一个工具时，这个方法会执行。它检查被调用的工具名：

- 如果是 `calculate_tip`（动态注册的），告诉 Agent 用 `calculate_tip` 函数来执行
- 如果是其他工具（静态注册的），正常处理（`return handler(request)`）

**为什么需要 `wrap_tool_call`？**因为动态工具不在 Agent 的原始工具列表中。当模型返回"我要调用 calculate_tip"时，Agent 不知道这个工具对应哪个函数。`wrap_tool_call` 就是告诉 Agent"遇到这个工具名时，用这个函数来执行"。

**方式一 vs 方式二的选择：**

|                  | 过滤预注册工具               | 运行时注册工具                                |
| ---------------- | ---------------------------- | --------------------------------------------- |
| 工具是否提前已知 | 是                           | 不一定                                        |
| 使用场景         | 权限控制、功能开关           | MCP 服务器、远程工具注册表                    |
| 实现复杂度       | 简单（只需 wrap_model_call） | 较高（需要 wrap_model_call + wrap_tool_call） |

## 03.4 工具错误处理

在生产环境中，工具调用可能失败（网络超时、参数错误等）。可以用 `@wrap_tool_call` 来统一处理错误：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4.1",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

**逐行详解：**

**`try: return handler(request)`**，尝试正常执行工具。`handler(request)` 就是"真正去执行工具"

**`except Exception as e:`**，如果工具执行过程中抛出任何异常（比如 `ZeroDivisionError`、`TimeoutError` 等），就进入这个错误处理分支

**`ToolMessage(...)`**，返回一个 `ToolMessage` 对象。这是 LangChain 定义的消息类型，专门用于表示工具的返回结果。参数说明：

- `content`：错误信息。注意这段话是给**模型**看的，不是给用户看的。模型收到这个错误信息后，可能会决定重试或换一种方式处理。
- `tool_call_id`：与模型发出的工具调用请求对应的 ID。模型可能同时调用多个工具，这个 ID 帮助模型把错误和对应的调用匹配起来。

**没有错误处理会怎样？**如果不设置错误处理中间件，工具抛出的异常会直接传播，可能导致整个 Agent 崩溃。设置了之后，异常会被捕获并转换为友好的错误消息，模型可以据此做出下一步决策

# 04 核心组件三：System Prompt

## 04.1 作用

没有系统提示词也可以运行，但Agent的行为会很"随意"。它不知道自己应该扮演什么角色、用什么语气、遵循什么规则。就像一个大堂经理没有培训就上岗，他可能有时很热情、有时很冷淡、有时说太多废话、有时又太简短

## 04.2 基本用法

`system_prompt` 参数接收一个字符串，定义 Agent 的角色和行为准则。这段话会在每次对话开始时发送给模型。如果不提供 `system_prompt`，Agent 会直接从用户消息中推断任务——这对简单任务可行，但对复杂场景不推荐

```python
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

## 04.3 SystemMessage方式

```python
from langchain.messages import SystemMessage

literary_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type": "text",
                "text": "<整本《傲慢与偏见》的全文>",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    )
)
```

**为什么要用 `SystemMessage` 而不是简单字符串？**`SystemMessage` 允许你传入一个**内容块列表**（而不是单一字符串），每个块可以有额外的配置

**`cache_control` 是什么？**

```python
"cache_control": {"type": "ephemeral"}
```

假设系统提示词包含一整本书（约 10 万字）。每次用户发消息时，Agent 都需要把这 10 万字的系统提示词发送给模型 API。这意味着：

- **每次请求**都要传输 10 万字 → 网络延迟
- **每次请求**都要为 10 万字付费 → 成本翻倍

`cache_control: {"type": "ephemeral"}` 告诉 Anthropic："请在你那边缓存这段内容。下次我发请求时，你直接从缓存读取，不需要我重新传。"

这样：

- 第 1 次请求：完整传输 10 万字（正常速度、正常价格）
- 第 2~N 次请求：不需要传这 10 万字了（更快、更便宜）

## 04.4 动态系统提示词

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role", "user")

    if user_role == "expert":
        return "You are a helpful assistant. Provide detailed technical responses."
    elif user_role == "beginner":
        return "You are a helpful assistant. Explain concepts simply and avoid jargon."
    return "You are a helpful assistant."
```

**`@dynamic_prompt` 装饰器：**是个中间件装饰器。它把函数变成一个"提示词生成器"——每次模型被调用时，Agent 会先执行函数来生成系统提示词



**`request.runtime.context.get("user_role", "user")`：**从运行时上下文中获取 `user_role`。注意这里用了 `TypedDict` 而不是 `@dataclass`。`TypedDict` 是 Python 的类型提示工具，定义的是字典的结构（键名和值类型），实际使用时就是普通字典。所以调用时传的 `context={"user_role": "expert"}`（一个字典），而不是 `context=Context(user_role="expert")`（一个对象）

**效果：**

- 专家用户问"解释机器学习"→ 系统提示词是"...提供详细的技术性回答"→ 模型给出深入的技术解释
- 初学者问同样的问题 → 系统提示词是"...用简单的语言解释，避免行话"→ 模型给出通俗易懂的解释

## 04.5 Agent命名

```python
agent = create_agent(
    model,
    tools,
    name="research_assistant"
)
```

`name` 参数给 Agent 设置一个标识名。主要用途是在**多 Agent 系统**中区分不同的 Agent。当你把一个 Agent 作为"子图"嵌入到更大的系统中时，这个名字就是它的节点标识符

**命名规范：** 使用 `snake_case`（如 `research_assistant`），不要用空格或特殊字符（如 `Research Assistant`），因为某些模型提供商不支持

# 05 Agent 的调用方式（Invocation）

## 05.1 基本调用

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```

**传入的参数是 Agent 的"状态（State）"：**

Agent 内部维护一个状态对象。最基本的状态包含 `messages`——一个消息列表。你通过传入新消息来触发 Agent 的处理流程。

**`result` 返回的是什么？**

`result` 也是一个状态字典，包含处理后的完整消息列表。你可以从中获取 Agent 的回答：

```python
# 获取最后一条消息（通常就是 Agent 的回答）
answer = result["messages"][-1].content

# 如果使用了结构化输出
structured = result["structured_response"]
```

## 05.2 对话格式说明

消息列表中的每条消息都有 `role` 和 `content`：

```python
{"messages": [
    {"role": "user", "content": "你好"},                    # 用户说的话
    {"role": "assistant", "content": "你好！有什么能帮你？"},  # AI 的回复
    {"role": "user", "content": "今天天气怎么样？"},          # 用户的追问
]}
```

一般来说，你只需要传最新的用户消息。如果你配置了 `checkpointer`（记忆组件），Agent 会自动从记忆中加载之前的对话历史

# 06 高级概念一：结构化输出

## 06.1 没有结构化输出的问题

让 Agent 提取联系人信息。没有结构化输出时：

```python
result = agent.invoke({"messages": [
    {"role": "user", "content": "Extract: John Doe, john@example.com, (555) 123-4567"}
]})

# 模型可能返回这些格式中的任何一种：
# "联系人是 John Doe，邮箱 john@example.com，电话 (555) 123-4567。"
# "Name: John Doe\nEmail: john@example.com\nPhone: (555) 123-4567"
# "我提取到了以下信息：John Doe 的联系方式是..."
```

代码需要从这些**不可预测的自由文本**中提取出名字、邮箱、电话。这非常困难，也非常脆弱——模型稍微换个说法，解析逻辑就崩了

## 06.2 有结构化输出后的幸福

```python
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4.1-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({"messages": [
    {"role": "user", "content": "Extract: John Doe, john@example.com, (555) 123-4567"}
]})

contact = result["structured_response"]
print(contact.name)    # 确定是 "John Doe"
print(contact.email)   # 确定是 "john@example.com"
print(contact.phone)   # 确定是 "(555) 123-4567"
```

**输出格式是确定的、可靠的、可编程的。*

## 06.3 ToolStrategy 的工作原理

`ToolStrategy` 的原理非常巧妙。一步步拆解：

```
步骤 1：你定义了 ContactInfo 这个数据模型

步骤 2：ToolStrategy(ContactInfo) 把 ContactInfo 伪装成一个"工具"
         这个工具的描述大概是：
         ┌──────────────────────────────────────┐
         │ 工具名：ContactInfo                   │
         │ 描述：提供最终的结构化回答              │
         │ 参数：                                │
         │   - name (string, 必填)               │
         │   - email (string, 必填)              │
         │   - phone (string, 必填)              │
         └──────────────────────────────────────┘

步骤 3：当 Agent 运行时，模型除了看到你注册的真实工具外，
         还会看到这个"假工具" ContactInfo

步骤 4：当模型完成所有推理后，它不会直接输出文本，
         而是"调用"ContactInfo 这个工具：
         {
           "tool": "ContactInfo",
           "arguments": {
             "name": "John Doe",
             "email": "john@example.com",
             "phone": "(555) 123-4567"
           }
         }

步骤 5：LangChain 截获这个"工具调用"，
         把 arguments 转换成 ContactInfo 对象，
         存入 result["structured_response"]
```

**为什么这种方式很可靠？**

因为模型调用工具时，**必须严格按照工具定义的参数格式传参**——这是所有支持工具调用的模型都遵守的规则。所以通过把输出格式"伪装"成工具，就借用了模型"严格遵循工具参数格式"这个已有的能力

## 06.4 ProviderStrategy——另一种方式

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4.1",
    response_format=ProviderStrategy(ContactInfo)
)
```

`ProviderStrategy` 不是伪装成工具，而是直接使用模型提供商的**原生结构化输出能力**。比如 OpenAI 提供了一个叫 "JSON Mode" 的功能，可以保证模型输出合法的 JSON

**什么时候用哪个？**

```
                         模型支持原生结构化输出？
                              /        \
                            是           否
                            /              \
                    ProviderStrategy    ToolStrategy
                    （更可靠）          （通用方案）
```

**自动选择：** 在 LangChain 1.0 中，你可以直接传 `response_format=ContactInfo`，LangChain 会自动判断用哪种策略

# 07 高级概念二：Memory & State

## 07.1 短期记忆——消息历史

Agent 通过维护消息列表来实现短期记忆。每次模型调用时，之前的所有消息（用户的、Agent 的、工具返回的）都会一起发送给模型，让模型"记住"之前的对话

## 07.2 自定义状态——存储额外信息

有时你需要在状态中存储消息以外的信息，比如用户偏好、工具调用统计等。**关键规则：** 自定义状态必须是 `TypedDict` 类型，并且必须继承 `AgentState`

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    user_preferences: dict
```

`AgentState` 是 LangChain 定义的基础状态类型，包含 `messages` 字段。你继承它，添加自己的字段

**两种定义方式：**

### 1 方式一：通过中间件定义（推荐）

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState         # 声明使用自定义状态
    tools = [tool1, tool2]             # 这个中间件关联的工具

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # 在模型调用前，可以访问和修改自定义状态
        preferences = state["user_preferences"]
        # ... 做一些处理
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

**`state_schema = CustomState`：** 在中间件类中声明状态格式。这样 LangChain 知道状态中除了 `messages`，还有 `user_preferences`

**`before_model` 方法：** 这是中间件的一个钩子（hook），在每次模型调用之前执行。可以在这里读取自定义状态并做处理

**为什么推荐这种方式？** 因为它把状态扩展和使用状态的逻辑放在同一个中间件中，代码组织更清晰

### 2 方式二：通过 state_schema 参数定义

```python
class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState    # 直接传给 create_agent
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

这种方式更简单，但自定义状态只能在工具中访问，不能在中间件的 `before_model` 等钩子中使用

# 08 高级概念三：流式输出（Streaming）

## 08.1 invoke vs stream

```css
invoke 模式（一次性返回）:

用户发消息 ─────────────────────────────────→ 等待 10 秒 ─→ 一次性收到完整回答
              （这 10 秒里用户什么都看不到）


stream 模式（边处理边返回）:

用户发消息 ─→ 看到"正在搜索..."
             ─→ 看到"找到 5 个结果"
             ─→ 看到"正在分析..."
             ─→ 看到最终回答
              （用户一直能看到进度）
```

## 08.2 流式输出的用法

```python
from langchain.messages import AIMessage, HumanMessage

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]},
    stream_mode="values"
):
    latest_message = chunk["messages"][-1]

    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

**`agent.stream(..., stream_mode="values")`**，`stream()` 方法和 `invoke()` 接收相同的参数（消息、配置等），但返回的不是最终结果，而是一个**迭代器**——每次 Agent 完成一个步骤，就产出（yield）一个 `chunk`



`stream_mode="values"` 表示每个 chunk 包含当前的完整状态（所有消息），而不是增量更新



**`for chunk in agent.stream(...):`，**Python 的 `for` 循环会自动从迭代器中逐个获取 chunk。每当一个新步骤完成，循环体就执行一次



**`chunk["messages"][-1]`**，获取当前状态中最新的一条消息。因为 `stream_mode="values"` 下每个 chunk 是完整状态，最后一条消息就是最新产生的

**消息类型判断：**

```python
if latest_message.content:
    # 消息有文本内容
    if isinstance(latest_message, HumanMessage):
        print(f"User: {latest_message.content}")
    elif isinstance(latest_message, AIMessage):
        print(f"Agent: {latest_message.content}")
elif latest_message.tool_calls:
    # 消息是工具调用请求（没有文本内容，但有工具调用信息）
    print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

`isinstance(latest_message, AIMessage)` 检查消息是否是 AI 生成的。LangChain 定义了多种消息类型：

| 消息类型        | 含义           | 关键属性                                               |
| --------------- | -------------- | ------------------------------------------------------ |
| `HumanMessage`  | 用户发送的消息 | `content`（文本内容）                                  |
| `AIMessage`     | 模型生成的消息 | `content`（文本）、`tool_calls`（工具调用列表）        |
| `ToolMessage`   | 工具执行结果   | `content`（结果文本）、`tool_call_id`（对应的调用 ID） |
| `SystemMessage` | 系统指令       | `content`（指令文本）                                  |

**`latest_message.tool_calls`：**

当模型决定调用工具时，它返回的 `AIMessage` 不包含 `content`（没有文本），而是包含 `tool_calls`——一个工具调用列表。每个元素是一个字典：

```python
{
    "name": "search",               # 工具名
    "args": {"query": "AI news"},   # 传给工具的参数
    "id": "call_abc123"             # 调用 ID
}
```

**流式输出的实际效果：**

```
User: Search for AI news and summarize the findings
Calling tools: ['search']
Agent: I found several interesting articles about AI...
```

# 09 高级概念四：中间件（Middleware）

## 09.1 什么是中间件？

中间件是"插入到 Agent 执行流程中的自定义逻辑"。你可以把它想象成一条流水线上的检查站：

```
用户输入 → [中间件 A] → 模型处理 → [中间件 B] → 工具调用 → [中间件 C] → 最终输出
```

每个中间件可以：

- **查看**经过它的数据
- **修改**经过它的数据
- **拦截**数据（比如阻止不安全的工具调用）

## 09.2 中间件的类型（按执行时机分）

文档中提到了多种中间件装饰器和方法，对应不同的执行时机：

| 装饰器 / 方法      | 执行时机             | 典型用途                         |
| ------------------ | -------------------- | -------------------------------- |
| `@before_model`    | 模型调用**之前**     | 消息裁剪、上下文注入、日志记录   |
| `@after_model`     | 模型调用**之后**     | 内容审核、结果验证、安全过滤     |
| `@wrap_model_call` | **包裹**整个模型调用 | 动态模型选择、重试逻辑、性能监控 |
| `@wrap_tool_call`  | **包裹**工具调用     | 错误处理、工具调用日志、权限检查 |
| `@dynamic_prompt`  | 模型调用**之前**     | 动态生成系统提示词               |

## 09.3 中间件的两种定义方式

### 1 方式一：装饰器（简单场景）

当你只需要拦截一个环节时：

```python
@wrap_model_call
def my_middleware(request: ModelRequest, handler) -> ModelResponse:
    # 在模型调用前做一些事
    print("About to call model...")
    response = handler(request)   # 调用模型
    # 在模型调用后做一些事
    print("Model responded!")
    return response
```

### 2 方式二：类继承（复杂场景）

当你需要拦截多个环节、或者需要维护自己的状态时：

```python
class MyMiddleware(AgentMiddleware):
    state_schema = CustomState     # 可选：自定义状态格式
    tools = [my_tool]              # 可选：中间件关联的工具

    def wrap_model_call(self, request, handler):
        # 拦截模型调用
        ...

    def wrap_tool_call(self, request, handler):
        # 拦截工具调用
        ...

    def before_model(self, state, runtime):
        # 在模型调用前修改状态
        ...

    def after_model(self, state, runtime):
        # 在模型调用后修改状态
        ...
```

## 09.4 中间件的实际应用场景汇总

| 场景           | 使用的钩子         | 做什么                                              |
| -------------- | ------------------ | --------------------------------------------------- |
| 消息裁剪       | `@before_model`    | 对话太长时，截取最近的 N 条消息，防止超出上下文窗口 |
| 内容审核       | `@after_model`     | 检查模型回答是否包含不安全内容，如果有就拦截        |
| 动态模型选择   | `@wrap_model_call` | 根据对话复杂度选择不同的模型                        |
| 动态系统提示词 | `@dynamic_prompt`  | 根据用户角色生成不同的提示词                        |
| 工具错误处理   | `@wrap_tool_call`  | 捕获工具异常，返回友好的错误信息                    |
| 权限控制       | `@wrap_model_call` | 根据用户权限过滤可用工具                            |
| 性能监控       | `@wrap_model_call` | 记录每次模型调用的耗时                              |
| 日志记录       | 任何钩子           | 记录 Agent 的执行过程，用于调试                     |

## 09.5 `ModelRequest` 对象详解

中间件函数中的 `request: ModelRequest` 是一个包含所有请求信息的对象。主要属性：

| 属性                      | 类型                  | 说明                                         |
| ------------------------- | --------------------- | -------------------------------------------- |
| `request.state`           | dict                  | Agent 当前状态（包含 messages 和自定义字段） |
| `request.tools`           | list                  | 当前可用的工具列表                           |
| `request.runtime`         | object                | 运行时信息                                   |
| `request.runtime.context` | 你定义的 Context 类型 | 运行时上下文（用户 ID、角色等）              |
| `request.runtime.store`   | Store 对象            | 持久化存储（用于跨会话数据）                 |
| `request.override(...)`   | method                | 创建修改后的请求副本                         |

`request.override()` 是一个非常重要的方法。它不会修改原始请求，而是创建一个新的请求对象，其中指定的字段被替换：

```python
# 替换模型
new_request = request.override(model=advanced_model)

# 替换工具列表
new_request = request.override(tools=filtered_tools)

# 同时替换多个字段
new_request = request.override(model=advanced_model, tools=filtered_tools)
```

# 10 完整场景例子

### 场景描述

你在做一个**企业知识库 AI 助手**。需求是：

- 根据用户角色（admin/viewer）提供不同的能力
- 根据用户角色使用不同的系统提示词
- 根据对话复杂度自动选择模型
- 工具调用失败时优雅处理
- 返回结构化输出
- 保存对话历史

### 完整代码

```python
from dataclasses import dataclass
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import (
    wrap_model_call, wrap_tool_call, dynamic_prompt,
    ModelRequest, ModelResponse
)
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# ===== 1. 定义上下文 =====
@dataclass
class Context:
    user_id: str
    user_role: str     # "admin" 或 "viewer"

# ===== 2. 定义工具 =====
@tool
def read_document(doc_id: str) -> str:
    """读取公司文档内容"""
    return f"文档 {doc_id} 的内容：[这里是文档内容]"

@tool
def delete_document(doc_id: str) -> str:
    """删除公司文档（仅管理员可用）"""
    return f"文档 {doc_id} 已被删除"

# ===== 3. 定义输出格式 =====
class AssistantResponse(BaseModel):
    answer: str                          # 回答内容
    source_documents: list[str] | None = None  # 引用的文档列表
    confidence: float = 1.0              # 置信度

# ===== 4. 动态系统提示词 =====
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    role = request.runtime.context.user_role
    if role == "admin":
        return "你是公司知识库管理员助手。你可以读取和删除文档。"
    else:
        return "你是公司知识库查询助手。你只能帮助用户查找信息。"

# ===== 5. 动态模型选择 =====
basic = ChatOpenAI(model="gpt-4.1-mini")
advanced = ChatOpenAI(model="gpt-4.1")

@wrap_model_call
def smart_model_selection(request: ModelRequest, handler) -> ModelResponse:
    # 根据消息数量和用户角色选择模型
    msg_count = len(request.state["messages"])
    role = request.runtime.context.user_role

    if msg_count > 10 or role == "admin":
        model = advanced    # 长对话或管理员 → 用强模型
    else:
        model = basic       # 短对话的普通用户 → 用省钱模型

    # 根据用户角色过滤工具
    if role != "admin":
        tools = [t for t in request.tools if t.name != "delete_document"]
        request = request.override(tools=tools, model=model)
    else:
        request = request.override(model=model)

    return handler(request)

# ===== 6. 错误处理 =====
@wrap_tool_call
def handle_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"操作失败：{str(e)}。请稍后重试。",
            tool_call_id=request.tool_call["id"]
        )

# ===== 7. 组装 Agent =====
agent = create_agent(
    model=basic,                                    # 默认模型
    tools=[read_document, delete_document],          # 所有工具
    middleware=[
        role_based_prompt,           # 先生成提示词
        smart_model_selection,       # 再选择模型和过滤工具
        handle_errors,               # 错误处理
    ],
    context_schema=Context,
    response_format=ToolStrategy(AssistantResponse),
    checkpointer=InMemorySaver(),                   # 对话记忆
)

# ===== 8. 运行 =====
# 普通用户查询文档
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查一下文档 DOC-001 的内容"}]},
    config={"configurable": {"thread_id": "user-123"}},
    context=Context(user_id="123", user_role="viewer"),
)
print(result["structured_response"])
```

 这段代码运行时的完整流程：

```
用户："帮我查一下文档 DOC-001 的内容"
上下文：user_role = "viewer"
         │
         ▼
1. 加载对话历史
   thread_id = "user-123"
   checkpointer 中无历史 → 空白开始
         │
         ▼
2. 中间件按顺序执行：

   ①  role_based_prompt 执行：
       user_role = "viewer"
       → 系统提示词 = "你是公司知识库查询助手。你只能帮助用户查找信息。"

   ②  smart_model_selection 执行：
       msg_count = 1（刚开始对话）
       user_role = "viewer"（不是 admin）
       → 模型选择 basic（gpt-4.1-mini）
       → 工具过滤：删除 delete_document
       → 模型现在只能看到 [read_document]

   ③  handle_errors 准备就绪（等待工具调用时生效）
         │
         ▼
3. 模型收到的信息：
   ┌──────────────────────────────────────────┐
   │ 系统提示词（动态生成的）：                  │
   │   "你是公司知识库查询助手。你只能帮助       │
   │    用户查找信息。"                         │
   │                                          │
   │ 可用工具（过滤后的）：                     │
   │   read_document(doc_id: str)             │
   │   （delete_document 被隐藏了！）           │
   │                                          │
   │ 用户消息：                                │
   │   "帮我查一下文档 DOC-001 的内容"          │
   └──────────────────────────────────────────┘
         │
         ▼
4. 模型决定调用 read_document(doc_id="DOC-001")
         │
         ▼
5. handle_errors 拦截工具调用：
   try → handler 执行 read_document("DOC-001")
   → 成功返回"文档 DOC-001 的内容：[这里是文档内容]"
   → 没有异常，正常返回
         │
         ▼
6. 模型收到工具结果，生成结构化输出：
   调用 AssistantResponse "工具"：
   {
       "answer": "文档 DOC-001 的内容如下：[这里是文档内容]",
       "source_documents": ["DOC-001"],
       "confidence": 0.95
   }
         │
         ▼
7. LangChain 把结果转换为 AssistantResponse 对象
         │
         ▼
8. 对话历史保存到 InMemorySaver（thread_id="user-123"）
         │
         ▼
9. 返回结果给你的代码：
   result["structured_response"] =
       AssistantResponse(
           answer="文档 DOC-001 的内容如下：...",
           source_documents=["DOC-001"],
           confidence=0.95
       )
```

---

# 11 速查手册

## 核心 API 一览

| 你想做什么     | 用什么                              | 怎么用                                               |
| -------------- | ----------------------------------- | ---------------------------------------------------- |
| 创建 Agent     | `create_agent()`                    | `agent = create_agent(model, tools)`                 |
| 运行 Agent     | `agent.invoke()`                    | `result = agent.invoke({"messages": [...]})`         |
| 流式运行       | `agent.stream()`                    | `for chunk in agent.stream({...}):`                  |
| 定义工具       | `@tool`                             | `@tool` 装饰你的函数                                 |
| 配置模型       | `ChatOpenAI()` 等                   | `model = ChatOpenAI(model="gpt-5", temperature=0.1)` |
| 结构化输出     | `ToolStrategy` / `ProviderStrategy` | `response_format=ToolStrategy(MySchema)`             |
| 对话记忆       | `InMemorySaver`                     | `checkpointer=InMemorySaver()`                       |
| 拦截模型调用   | `@wrap_model_call`                  | 装饰一个函数，修改请求或换模型                       |
| 拦截工具调用   | `@wrap_tool_call`                   | 装饰一个函数，处理错误或路由工具                     |
| 动态提示词     | `@dynamic_prompt`                   | 装饰一个函数，根据条件返回不同提示词                 |
| 自定义状态     | `AgentState` 子类                   | 定义 `TypedDict`，继承 `AgentState`                  |
| 模型调用前处理 | `@before_model`                     | 裁剪消息、注入上下文                                 |
| 模型调用后处理 | `@after_model`                      | 内容审核、结果验证                                   |
| 运行时上下文   | `context_schema` + `Context`        | 把用户信息传给工具和中间件                           |

## 核心概念关系图

```css
create_agent（工厂函数）
│
├── model（大脑）
│   ├── 字符串方式："gpt-5"
│   └── 实例方式：ChatOpenAI(model="gpt-5", temperature=0.1)
│
├── tools（手和脚）
│   ├── @tool 装饰的函数
│   └── 模型根据名字+描述+参数来决定何时调用
│
├── system_prompt（规矩）
│   ├── 字符串
│   └── SystemMessage（支持缓存等高级功能）
│
├── middleware（插件/检查站）
│   ├── @before_model ── 模型调用前
│   ├── @dynamic_prompt ── 动态生成提示词
│   ├── @wrap_model_call ── 包裹模型调用（换模型、过滤工具）
│   ├── @after_model ── 模型调用后
│   └── @wrap_tool_call ── 包裹工具调用（错误处理）
│
├── response_format（输出格式）
│   ├── ToolStrategy（通用方案，伪装成工具）
│   └── ProviderStrategy（原生方案，更可靠）
│
├── checkpointer（记忆）
│   └── InMemorySaver / 数据库保存器
│
├── context_schema（运行时上下文）
│   └── 你定义的 Context 类（包含 user_id、user_role 等）
│
└── state_schema（自定义状态）
    └── 继承 AgentState 的 TypedDict
```

