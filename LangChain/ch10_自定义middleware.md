# LangChain 自定义中间件（Custom Middleware）深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第九节。上一节（第八节）我们学了 LangChain 提供的 16 个预置中间件，知道了中间件能做什么。本节学习**如何自己动手写中间件**——理解钩子机制、掌握两种创建方式、学会状态管理和执行控制。如果说第八节是"学会用别人造的轮子"，那本节是"学会自己造轮子"

---

## 第一章：回顾——中间件在哪里执行？

在动手写之前，先精确理解中间件的**六个插入点**

### 1.1 Agent 循环中的六个钩子

```
Agent 执行流程（每一轮 ReAct 循环）:

  ┌─ before_agent ─────────────────────── 整个调用开始前（只执行一次）
  │
  │   ┌─ before_model ──────────────── 每次模型调用前
  │   │
  │   │   ┌─ wrap_model_call ─────── 包裹模型调用（可重试/替换/短路）
  │   │   │     ↓
  │   │   │   模型执行
  │   │   │     ↓
  │   │   └─────────────────────────
  │   │
  │   └─ after_model ───────────────── 每次模型返回后
  │
  │   模型返回了工具调用？
  │     ├── 否 → 跳到 after_agent
  │     └── 是 ↓
  │
  │   ┌─ wrap_tool_call ──────────── 包裹工具调用（可重试/监控/短路）
  │   │     ↓
  │   │   工具执行
  │   │     ↓
  │   └─────────────────────────────
  │
  │   回到 before_model ↑（下一轮循环）
  │
  └─ after_agent ──────────────────────── 整个调用结束后（只执行一次）
```

### 1.2 两类钩子的本质区别

| 特性 | Node-style 钩子 | Wrap-style 钩子 |
|:----:|:--------------:|:--------------:|
| 钩子名 | `before_agent` / `before_model` / `after_model` / `after_agent` | `wrap_model_call` / `wrap_tool_call` |
| 执行方式 | 在特定时间点**顺序执行** | **包裹**实际调用，像洋葱皮一层层嵌套 |
| 能控制执行吗？ | 不能——只能观察和修改状态 | 能——决定是否调用、调用几次、用什么参数 |
| 典型用途 | 日志、验证、状态更新 | 重试、缓存、模型替换、工具监控 |

**用一个生活类比理解：**

```
Node-style 像"检查站":
  你经过检查站 → 检查员看一眼你的证件 → 放行或拦截
  检查员不能替你开车，只能决定让不让你过

Wrap-style 像"代理人":
  你把任务交给代理人 → 代理人决定怎么执行
  代理人可以：
    - 直接执行（正常流程）
    - 执行失败后重试（重试逻辑）
    - 换一种方式执行（替换模型/工具）
    - 不执行直接返回（短路/缓存）
    - 执行多次取最好的结果
```

---

## 第二章：两种创建方式——装饰器 vs 类

LangChain 提供两种方式创建自定义中间件：**装饰器**（快速简洁）和**类**（功能完整）

### 2.1 装饰器方式——适合简单场景

每个装饰器对应一个钩子，一个函数就是一个中间件：

```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langgraph.runtime import Runtime
from typing import Any


# 装饰器方式：一个函数 = 一个中间件
@before_model
def log_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """在每次模型调用前打印消息数量。"""
    print(f"即将调用模型，当前有 {len(state['messages'])} 条消息")
    return None  # 返回 None 表示不修改状态


@after_model
def log_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """在每次模型返回后打印结果。"""
    print(f"模型返回: {state['messages'][-1].content}")
    return None
```

**使用：**

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[log_before, log_after],  # 直接传入装饰过的函数
)
```

### 2.2 类方式——适合复杂场景

一个类可以包含多个钩子，还能有配置参数和同步/异步两套实现：

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any


class LoggingMiddleware(AgentMiddleware):
    """日志中间件——记录模型调用前后的信息。"""

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose  # 可配置参数

    # 同步版本
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"即将调用模型，当前有 {len(state['messages'])} 条消息")
        if self.verbose:
            for msg in state["messages"][-3:]:
                print(f"  最近消息: {msg.content[:50]}...")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"模型返回: {state['messages'][-1].content[:100]}")
        return None

    # 异步版本（可选——用于 astream/ainvoke）
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # 异步版本的逻辑，可以用 await 调用异步操作
        return self.before_model(state, runtime)

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
```

**使用：**

```python
agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[LoggingMiddleware(verbose=True)],  # 实例化后传入
)
```

### 2.3 选择哪种方式？

| 场景 | 推荐方式 | 原因 |
|:----:|:-------:|:----:|
| 只需要一个钩子 | 装饰器 | 代码最简洁 |
| 不需要配置参数 | 装饰器 | 没有 `__init__` 的开销 |
| 快速原型验证 | 装饰器 | 写完即用 |
| 需要多个钩子配合 | 类 | 一个类里放多个钩子方法 |
| 需要配置参数 | 类 | 通过 `__init__` 传参 |
| 需要同步+异步实现 | 类 | 类可以同时定义 `before_model` 和 `abefore_model` |
| 跨项目复用 | 类 | 封装更好，接口更清晰 |

---

## 第三章：Node-style 钩子详解

### 3.1 四个钩子的函数签名

所有 Node-style 钩子的签名都是一样的：

```python
def hook_name(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    参数:
        state: 当前的 Agent 状态（包含 messages 等字段）
        runtime: 运行时上下文（包含配置信息等）

    返回:
        dict: 要合并到 Agent 状态中的更新（通过 reducer 合并）
        None: 不修改状态
    """
```

### 3.2 `before_agent` / `after_agent`——Agent 级别（每次调用只执行一次）

```python
from langchain.agents.middleware import before_agent, after_agent, AgentState
from langgraph.runtime import Runtime
from typing import Any
import time


@before_agent
def start_timer(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """记录 Agent 开始执行的时间。"""
    print(f"Agent 开始执行，时间: {time.time()}")
    return None


@after_agent
def end_timer(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Agent 执行完毕。"""
    print(f"Agent 执行完毕，时间: {time.time()}")
    return None
```

**执行时机：**

```
agent.invoke({"messages": [...]})
  ↓
before_agent()     ← 只执行一次
  ↓
[Agent 循环: before_model → 模型 → after_model → 工具 → ...]
[Agent 循环: before_model → 模型 → after_model → ...]
  ↓
after_agent()      ← 只执行一次
```

### 3.3 `before_model` / `after_model`——模型调用级别（每轮循环都执行）

```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


@before_model(can_jump_to=["end"])  # ← 声明可能跳转到 "end"
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """如果消息过多，提前结束。"""
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("对话消息数已达上限。")],
            "jump_to": "end"  # ← 跳转指令
        }
    return None


@after_model
def log_model_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """记录模型每次返回的内容。"""
    last_msg = state["messages"][-1]
    print(f"模型返回: {last_msg.content[:100]}")
    return None
```

**执行时机：**

```
Agent 第一轮循环:
  before_model()   ← 执行
  模型调用
  after_model()    ← 执行
  工具调用

Agent 第二轮循环:
  before_model()   ← 又执行了
  模型调用
  after_model()    ← 又执行了
  结束
```

### 3.4 返回值的作用——状态更新

当 Node-style 钩子返回一个字典时，字典中的内容会被**合并到 Agent 状态**中：

```python
@after_model
def count_calls(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # 返回字典 → 更新 Agent 状态
    return {"model_call_count": state.get("model_call_count", 0) + 1}
    # 相当于: state["model_call_count"] += 1
```

**注意 `messages` 字段使用的是"追加"语义（additive reducer）：**

```python
@before_model
def inject_reminder(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    return {
        "messages": [SystemMessage("请用中文回答。")]
        # 不是替换所有 messages，而是追加一条新消息
    }
```

---

## 第四章：Wrap-style 钩子详解

### 4.1 `wrap_model_call`——包裹模型调用

Wrap-style 钩子接收两个参数：请求对象和处理函数（handler）。**你决定是否调用 handler 以及调用几次**

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """模型调用失败时自动重试，最多 3 次。"""
    for attempt in range(3):
        try:
            return handler(request)  # ← 调用实际的模型
        except Exception as e:
            if attempt == 2:  # 最后一次重试也失败了
                raise  # 抛出异常
            print(f"重试 {attempt + 1}/3，错误: {e}")
```

**`handler` 的本质：**

```
handler(request) 做了什么？

如果没有其他 wrap_model_call 中间件:
  handler = 实际调用模型的函数

如果有多层 wrap_model_call 中间件:
  handler = 下一层中间件的 wrap_model_call
  → 最内层的 handler 才是实际的模型调用

  middleware1.wrap_model_call:
    handler → middleware2.wrap_model_call:
                handler → middleware3.wrap_model_call:
                            handler → 实际模型调用
```

### 4.2 `wrap_tool_call`——包裹工具调用

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from typing import Callable


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """监控每次工具调用。"""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
    print(f"执行工具: {tool_name}, 参数: {tool_args}")

    try:
        result = handler(request)  # ← 调用实际的工具
        print(f"工具 {tool_name} 执行成功")
        return result
    except Exception as e:
        print(f"工具 {tool_name} 执行失败: {e}")
        raise
```

**`ToolCallRequest` 中的关键信息：**

```python
request.tool_call["name"]   # 工具名称
request.tool_call["args"]   # 工具参数
request.tool_call["id"]     # 调用 ID
```

### 4.3 Wrap-style 的四种经典模式

**模式一：正常放行**

```python
@wrap_model_call
def passthrough(request, handler):
    return handler(request)  # 什么都不做，直接调用
```

**模式二：修改请求**

```python
@wrap_model_call
def modify_request(request, handler):
    # 修改请求后再调用
    new_request = request.override(model=another_model)
    return handler(new_request)
```

**模式三：重试**

```python
@wrap_model_call
def retry(request, handler):
    for i in range(3):
        try:
            return handler(request)
        except:
            if i == 2: raise
```

**模式四：短路（不调用 handler）**

```python
@wrap_model_call
def cache(request, handler):
    cached = get_from_cache(request)
    if cached:
        return cached  # 直接返回缓存，不调用模型
    result = handler(request)
    save_to_cache(request, result)
    return result
```

### 4.4 `ModelRequest` 的 `override` 方法

`request.override()` 是修改请求的关键方法——它返回一个新的请求对象，不会修改原始请求：

```python
@wrap_model_call
def customize_request(request, handler):
    # 替换模型
    new_request = request.override(model=another_model)

    # 替换系统提示
    new_request = request.override(system_message=SystemMessage("新的系统提示"))

    # 替换工具列表
    new_request = request.override(tools=filtered_tools)

    # 可以链式调用
    return handler(
        request.override(model=another_model, tools=filtered_tools)
    )
```

**`ModelRequest` 的可用字段：**

```python
request.messages         # 对话消息列表
request.system_message   # 系统提示（始终是 SystemMessage 对象）
request.model            # 当前模型
request.tools            # 当前工具列表
request.state            # Agent 状态
request.runtime          # 运行时上下文
```

---

## 第五章：状态更新的高级用法

### 5.1 自定义状态 Schema

如果中间件需要在 Agent 状态中存储自己的数据（如调用计数器、用户 ID 等），需要**扩展 AgentState**：

```python
from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class CustomState(AgentState):
    """扩展的 Agent 状态——添加自定义字段。"""
    model_call_count: NotRequired[int]   # 模型调用计数
    user_id: NotRequired[str]            # 用户 ID
```

**`NotRequired` 的作用：** 标记这些字段是可选的。Agent 在 `invoke` 时不强制要求提供这些字段，但中间件可以读写它们。

### 5.2 在装饰器中使用自定义状态

通过 `state_schema` 参数告诉装饰器使用你的自定义状态类型：

```python
from langchain.agents.middleware import before_model, after_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langgraph.runtime import Runtime
from typing import Any


class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


@before_model(state_schema=CustomState, can_jump_to=["end"])
def check_limit(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
    """检查调用次数是否超限。"""
    count = state.get("model_call_count", 0)
    if count > 10:
        return {
            "messages": [AIMessage("调用次数已达上限。")],
            "jump_to": "end"
        }
    return None


@after_model(state_schema=CustomState)
def increment_counter(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
    """每次模型调用后，计数器加 1。"""
    return {"model_call_count": state.get("model_call_count", 0) + 1}


agent = create_agent(
    model="gpt-4.1",
    middleware=[check_limit, increment_counter],
    tools=[],
)

# 调用时可以传入自定义状态的初始值
result = agent.invoke({
    "messages": [HumanMessage("你好")],
    "model_call_count": 0,
    "user_id": "user-123",
})
```

### 5.3 在类中使用自定义状态

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from typing_extensions import NotRequired
from typing import Any


class CustomState(AgentState):
    model_call_count: NotRequired[int]


class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState  # ← 声明状态类型

    def __init__(self, max_calls: int = 10):
        super().__init__()
        self.max_calls = max_calls

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        if state.get("model_call_count", 0) > self.max_calls:
            return {"jump_to": "end"}
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}
```

### 5.4 Wrap-style 钩子的状态更新——ExtendedModelResponse

Node-style 钩子直接返回字典来更新状态。但 Wrap-style 钩子的返回值是模型响应（`ModelResponse`），不是字典。如果你也需要更新状态，要用 `ExtendedModelResponse`：

```python
from langchain.agents.middleware import (
    wrap_model_call, ModelRequest, ModelResponse, ExtendedModelResponse
)
from langgraph.types import Command
from typing import Callable


@wrap_model_call
def track_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    """调用模型并跟踪 token 使用量。"""
    response = handler(request)

    # 通过 ExtendedModelResponse + Command 来更新状态
    return ExtendedModelResponse(
        model_response=response,                      # 原始模型响应
        command=Command(update={                       # 状态更新
            "last_model_call_tokens": 150
        }),
    )
```

**为什么需要 `ExtendedModelResponse`？**

```
Node-style 钩子:
  返回 dict → 直接合并到状态
  简单直接

Wrap-style 钩子:
  必须返回 ModelResponse（因为下游需要模型的输出）
  但你也想更新状态...
  → ExtendedModelResponse 同时携带"模型响应"和"状态更新"
```

### 5.5 多个中间件的 Command 组合规则

当多个 Wrap-style 中间件都返回 `ExtendedModelResponse` 时：

```python
class OuterMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={
                "trace": "outer",                             # 非 reducer 字段
                "messages": [SystemMessage("[Outer 执行]")],  # messages 是 additive reducer
            }),
        )

class InnerMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={
                "trace": "inner",
                "messages": [SystemMessage("[Inner 执行]")],
            }),
        )
```

**组合规则：**

```
messages 字段（additive reducer）:
  → Inner 的消息 + Outer 的消息，都会被追加
  → 最终 messages 中同时包含 "[Inner 执行]" 和 "[Outer 执行]"

trace 字段（非 reducer / last-wins）:
  → Inner 先写 "inner"，Outer 后写 "outer"
  → 最终 trace = "outer"（外层中间件覆盖内层）

如果 Outer 有重试逻辑（多次调用 handler）:
  → 之前失败调用产生的 Command 会被丢弃
  → 只有最后一次成功调用的 Command 生效
```

---

## 第六章：Agent 跳转——提前结束或改变流程

### 6.1 `jump_to` 指令

在 Node-style 钩子中，返回字典时可以包含 `jump_to` 键来改变 Agent 的执行流程：

```python
@after_model(can_jump_to=["end"])  # ← 必须声明可能的跳转目标
def check_blocked(state, runtime):
    last_msg = state["messages"][-1]
    if "BLOCKED" in last_msg.content:
        return {
            "messages": [AIMessage("抱歉，我无法回答该请求。")],
            "jump_to": "end"  # ← 跳转到结束
        }
    return None
```

**可用的跳转目标：**

| 目标 | 效果 |
|------|------|
| `"end"` | 立即结束 Agent 执行（跳到 `after_agent` 钩子） |
| `"tools"` | 跳到工具执行节点 |
| `"model"` | 跳到模型调用节点（即 `before_model` 钩子） |

### 6.2 `can_jump_to` 声明

出于安全考虑，中间件必须**提前声明**它可能跳转到哪些目标。这样 LangChain 在构建 Agent 图时就知道哪些边是可能的：

```python
# 装饰器方式
@before_model(can_jump_to=["end"])
def my_hook(state, runtime):
    ...

# 或者用 hook_config 装饰器
@after_model
@hook_config(can_jump_to=["end", "model"])
def my_hook(state, runtime):
    ...

# 类方式
class MyMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        ...
```

**如果你返回了 `jump_to` 但没有用 `can_jump_to` 声明，会报错。**

### 6.3 跳转的典型场景

```python
# 场景一：对话过长，强制结束
@before_model(can_jump_to=["end"])
def limit_messages(state, runtime):
    if len(state["messages"]) > 100:
        return {"messages": [AIMessage("对话太长了，请开始新对话。")], "jump_to": "end"}
    return None

# 场景二：内容安全检查，拦截不当输出
@after_model(can_jump_to=["end"])
def content_filter(state, runtime):
    if is_harmful(state["messages"][-1].content):
        return {"messages": [AIMessage("检测到不当内容，已拦截。")], "jump_to": "end"}
    return None

# 场景三：跳过工具调用，直接让模型重新回答
@after_model(can_jump_to=["model"])
def skip_unnecessary_tools(state, runtime):
    last_msg = state["messages"][-1]
    if has_trivial_tool_calls(last_msg):
        return {"messages": [AIMessage("让我直接回答你。")], "jump_to": "model"}
    return None
```

---

## 第七章：执行顺序——多个中间件如何协作

### 7.1 完整的执行顺序规则

```python
agent = create_agent(
    model="gpt-4.1",
    middleware=[middleware1, middleware2, middleware3],
    tools=[...],
)
```

**规则一：`before_*` 钩子——按列表顺序执行（先 1 后 2 后 3）**

```
before_agent:
  middleware1.before_agent()  → middleware2.before_agent()  → middleware3.before_agent()

before_model:
  middleware1.before_model()  → middleware2.before_model()  → middleware3.before_model()
```

**规则二：`after_*` 钩子——按列表逆序执行（先 3 后 2 后 1）**

```
after_model:
  middleware3.after_model()  → middleware2.after_model()  → middleware1.after_model()

after_agent:
  middleware3.after_agent()  → middleware2.after_agent()  → middleware1.after_agent()
```

**规则三：`wrap_*` 钩子——嵌套执行（洋葱模型）**

```
wrap_model_call:
  middleware1.wrap_model_call:
    │ request → middleware2.wrap_model_call:
    │             │ request → middleware3.wrap_model_call:
    │             │             │ request → 实际模型调用
    │             │             │ ← response
    │             │ ← response
    │ ← response

  middleware1 是最外层，最先拿到 request，最后拿到 response
  middleware3 是最内层，最后拿到 request，最先拿到 response
```

### 7.2 用图示理解完整流程

```
时间 →

middleware1.before_agent()
  middleware2.before_agent()
    middleware3.before_agent()

    ┌─────────── Agent 循环 ───────────┐
    │                                   │
    │ middleware1.before_model()         │
    │   middleware2.before_model()       │
    │     middleware3.before_model()     │
    │                                   │
    │ middleware1.wrap_model_call(       │
    │   middleware2.wrap_model_call(     │
    │     middleware3.wrap_model_call(   │
    │       实际模型调用                 │
    │     )                             │
    │   )                               │
    │ )                                 │
    │                                   │
    │     middleware3.after_model()      │
    │   middleware2.after_model()        │
    │ middleware1.after_model()          │
    │                                   │
    └───────────────────────────────────┘

    middleware3.after_agent()
  middleware2.after_agent()
middleware1.after_agent()
```

**记忆口诀：** before 正序、after 倒序、wrap 嵌套。这和 Web 框架中间件的执行顺序完全一致（如 Express.js、Koa.js）。

---

## 第八章：实战案例

### 8.1 动态修改系统提示

在每次模型调用前动态注入上下文信息（如用户偏好、当前时间等）：

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable


@wrap_model_call
def add_dynamic_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """在系统提示中追加动态上下文。"""
    # content_blocks 把系统提示转为内容块列表（无论原始格式是字符串还是列表）
    new_content = list(request.system_message.content_blocks) + [
        {"type": "text", "text": "当前时间: 2026-04-12 14:30:00"}
    ]
    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))
```

**关键细节：**

- `request.system_message` **始终是 `SystemMessage` 对象**，即使 `create_agent` 时传的是字符串
- 使用 `content_blocks` 属性访问内容，它始终返回列表格式
- 追加新内容块时保留原有结构

### 8.2 动态选择模型

根据对话复杂度自动选择不同模型：

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# 预初始化两个模型
complex_model = init_chat_model("claude-sonnet-4-6")
simple_model = init_chat_model("claude-haiku-4-5-20251001")


@wrap_model_call
def dynamic_model_selector(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """消息多时用强模型，消息少时用快模型。"""
    if len(request.messages) > 10:
        model = complex_model
    else:
        model = simple_model
    return handler(request.override(model=model))
```

### 8.3 动态筛选工具

根据上下文只暴露相关工具给模型：

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def select_relevant_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """根据最后一条消息的内容筛选工具。"""
    last_message = request.messages[-1].content.lower()

    # 根据关键词筛选
    relevant = []
    for tool in request.tools:
        if "天气" in last_message and "weather" in tool.name:
            relevant.append(tool)
        elif "邮件" in last_message and "email" in tool.name:
            relevant.append(tool)
        # ... 更多规则

    if not relevant:
        relevant = request.tools  # 兜底：没匹配到就保留全部

    return handler(request.override(tools=relevant))
```

### 8.4 Anthropic 提示缓存

对于 Anthropic 模型，可以给大型系统提示添加缓存控制，避免每次调用都重新处理：

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable


@wrap_model_call
def add_cached_document(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """将大文档作为带缓存的系统提示注入。"""
    new_content = list(request.system_message.content_blocks) + [
        {
            "type": "text",
            "text": "这是一份需要分析的大型文档：\n\n<document>...10万字...</document>",
            "cache_control": {"type": "ephemeral"}  # ← Anthropic 缓存指令
            # 到这个内容块为止的所有内容都会被缓存
        }
    ]
    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))
```

### 8.5 完整的类中间件示例——内容安全护栏

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


class ContentGuardMiddleware(AgentMiddleware):
    """内容安全护栏——检查模型输出是否包含不当内容。"""

    def __init__(self, blocked_words: list[str] = None):
        super().__init__()
        self.blocked_words = blocked_words or ["暴力", "色情", "违法"]

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """模型调用前：记录输入。"""
        last_msg = state["messages"][-1]
        print(f"[Guard] 用户输入: {last_msg.content[:50]}...")
        return None

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """模型返回后：检查输出内容。"""
        last_msg = state["messages"][-1]
        content = last_msg.content if isinstance(last_msg.content, str) else ""

        for word in self.blocked_words:
            if word in content:
                print(f"[Guard] 检测到违禁词: {word}，拦截输出")
                return {
                    "messages": [AIMessage("抱歉，我无法提供该内容。")],
                    "jump_to": "end"
                }

        print("[Guard] 内容检查通过")
        return None


# 使用
agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[ContentGuardMiddleware(blocked_words=["敏感词A", "敏感词B"])],
)
```

---

## 第九章：概念串联

### 9.1 与前几节知识的关联

```
第一节（Agent）
  → 自定义中间件在 Agent 核心循环的六个点插入钩子
  → can_jump_to 声明允许中间件改变 Agent 的执行流

第二节（Model）
  → wrap_model_call 可以 override 模型（动态选择）
  → request.system_message 可以修改系统提示

第三节（Messages）
  → 钩子通过 state["messages"] 访问对话历史
  → 返回 {"messages": [...]} 时使用 additive reducer（追加而非替换）
  → AIMessage / SystemMessage / ToolMessage 都可以在钩子中构造

第四节（Tools）
  → wrap_tool_call 包裹每次工具调用
  → request.override(tools=...) 可以动态筛选工具
  → ToolCallRequest 包含工具名、参数、调用 ID

第七节（Structured Output）
  → LLMToolSelector 内部用 wrap_model_call 实现工具筛选
  → 自定义中间件也可以做类似的事

第八节（预置中间件）
  → 预置中间件是自定义中间件的"模板"
  → ModelRetryMiddleware 本质是 wrap_model_call + 重试逻辑
  → ModelFallbackMiddleware 本质是 wrap_model_call + override(model=...)
  → PIIMiddleware 本质是 before_model + 文本检测
  → 所有预置中间件都可以用本节的方式自己实现

本节（自定义中间件）← 你在这里
  → 六个钩子覆盖 Agent 执行的每个环节
  → 装饰器适合简单场景，类适合复杂场景
  → Node-style 观察和修改状态，Wrap-style 控制执行流程
  → 自定义状态 Schema 让中间件跨钩子共享数据
  → jump_to 实现提前结束或流程跳转
  → 执行顺序：before 正序、after 倒序、wrap 嵌套
```

---

## 第十章：速查手册

### 六个钩子速查

| 钩子 | 类型 | 执行频率 | 函数签名 |
|------|------|---------|---------|
| `before_agent` | Node | 每次调用 1 次 | `(state, runtime) → dict \| None` |
| `before_model` | Node | 每轮循环 1 次 | `(state, runtime) → dict \| None` |
| `after_model` | Node | 每轮循环 1 次 | `(state, runtime) → dict \| None` |
| `after_agent` | Node | 每次调用 1 次 | `(state, runtime) → dict \| None` |
| `wrap_model_call` | Wrap | 每次模型调用 | `(request, handler) → ModelResponse` |
| `wrap_tool_call` | Wrap | 每次工具调用 | `(request, handler) → ToolMessage \| Command` |

### 装饰器 vs 类 速查

| 特性 | 装饰器 | 类 |
|------|-------|-----|
| 代码量 | 少 | 多 |
| 钩子数量 | 一个函数一个钩子 | 一个类多个钩子 |
| 配置参数 | 不支持 | 通过 `__init__` |
| 异步支持 | 仅同步 | 同步 + 异步 |
| 适用场景 | 快速原型 | 生产环境 |

### 状态更新方式速查

| 钩子类型 | 更新方式 | 示例 |
|---------|---------|------|
| Node-style | 直接返回 dict | `return {"count": 1}` |
| Wrap-style | `ExtendedModelResponse` + `Command` | `return ExtendedModelResponse(response, Command(update={...}))` |

### 执行顺序速查

```
before_*:  middleware1 → middleware2 → middleware3     （正序）
after_*:   middleware3 → middleware2 → middleware1     （倒序）
wrap_*:    middleware1( middleware2( middleware3( 实际调用 )))  （嵌套）
```

### 跳转目标速查

| `jump_to` 值 | 效果 | 声明方式 |
|-------------|------|---------|
| `"end"` | 结束 Agent 执行 | `can_jump_to=["end"]` |
| `"tools"` | 跳到工具节点 | `can_jump_to=["tools"]` |
| `"model"` | 跳到模型节点 | `can_jump_to=["model"]` |

### `ModelRequest.override()` 可修改的字段

| 字段 | 说明 |
|------|------|
| `model` | 替换模型 |
| `system_message` | 替换系统提示 |
| `tools` | 替换工具列表 |
