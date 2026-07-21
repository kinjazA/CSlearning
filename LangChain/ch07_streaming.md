# LangChain 流式输出（Streaming）深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第六节。第一节学了 Agent（整体架构），第二节学了 Model（大脑），第三节学了 Messages（消息/纸条），第四节学了 Tools（手和脚），第五节学了 Short-term Memory（对话记忆），本节学习 Streaming——Agent 的"实时播报"能力。如果说前面五节解决了 Agent "能做什么"的问题，那流式输出解决的是"怎么让用户看到 Agent 在做什么"的问题

---

## 第一章：为什么需要流式输出？

### 1.1 没有流式输出的痛点

想象你在用一个 AI 助手，你问了一个问题。然后……

```
你: "帮我写一篇 500 字的文章"
AI: （加载中...）
AI: （加载中...）
AI: （加载中...）
   ...15 秒过去了...
AI: "这是一篇关于..."  ← 15 秒后突然全部出现
```

这 15 秒里你什么都看不到，不知道 AI 是在思考、在调用工具、还是卡住了。用户体验极差

**有了流式输出：**

```
你: "帮我写一篇 500 字的文章"
AI: "这" → "是" → "一" → "篇" → "关" → "于" → ...
     ↑ 每个字生成后立刻显示，用户几乎没有等待感
```

### 1.2 流式输出能传递什么？

LangChain 的流式系统不仅能传递文字——它能实时传递 Agent 运行过程中的各种信息：

```
┌─────────────────────────────────────────────────────┐
│  Agent 运行过程中的流式输出                           │
│                                                      │
│  ① Agent 进度（updates）                             │
│     "模型正在思考..." → "调用了 get_weather 工具"      │
│     → "工具返回了结果" → "模型生成最终回答"            │
│                                                      │
│  ② LLM Token（messages）                             │
│     "今" → "天" → "是" → "晴" → "天" → "，"          │
│     → "气" → "温" → "25" → "°C"                     │
│                                                      │
│  ③ 思考/推理过程（reasoning）                         │
│     "[思考] 用户问的是天气..."                         │
│     "[思考] 我需要调用天气工具..."                     │
│                                                      │
│  ④ 自定义更新（custom）                               │
│     "正在查询数据库..." → "已获取 50/100 条记录"       │
└─────────────────────────────────────────────────────┘
```

---

## 第二章：三种流式模式——你想看什么？

LangChain 提供三种流式模式（stream mode），每种模式传递不同粒度的信息。通过 `stream_mode` 参数来选择

### 2.1 模式总览

| 模式 | 传递内容 | 粒度 | 适用场景 |
|------|---------|------|---------|
| `"updates"` | 每一步的状态更新 | 步骤级 | 显示 Agent 进度、调试 |
| `"messages"` | LLM 生成的每个 token | Token 级 | 打字机效果、实时显示文字 |
| `"custom"` | 你自定义的任意数据 | 自定义 | 进度条、中间状态 |

**三种模式可以组合使用**——传入列表即可同时接收多种流：

```python
# 单一模式
agent.stream(..., stream_mode="updates")

# 组合模式
agent.stream(..., stream_mode=["updates", "messages", "custom"])
```

### 2.2 v2 流式格式

在深入三种模式之前，先了解 v2 格式。LangChain 推荐使用 `version="v2"` 来获得统一的输出格式：

```python
# v2 格式（推荐）——每个 chunk 都是统一的字典
for chunk in agent.stream(..., version="v2"):
    print(chunk["type"])  # 流的类型: "updates" / "messages" / "custom"
    print(chunk["data"])  # 实际数据
    print(chunk["ns"])    # 命名空间（用于子图区分）

# v1 格式（旧版默认）——多模式时需要元组解包
for mode, chunk in agent.stream(..., stream_mode=["updates", "custom"]):
    print(mode)   # "updates" 或 "custom"
    print(chunk)  # 数据
```

**为什么推荐 v2？**

v1 格式在单模式和多模式时的输出结构不一致（单模式直接返回数据，多模式返回元组），容易出错。v2 格式不管什么情况都返回相同结构的字典，代码更统一、更不容易出 bug

**v2 也改进了 `invoke()` 的返回值：**

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    version="v2",
)
print(result.value)       # Agent 的最终状态（dict 或 Pydantic 模型）
print(result.interrupts)  # 中断信息（如果有 human-in-the-loop）
```

---

## 第三章：Agent 进度流（updates 模式）

### 3.1 基本用法

`stream_mode="updates"` 在 Agent 每完成一步操作后发出一个事件。让你知道 Agent 当前执行到了哪一步

```python
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是晴天！"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

# 使用 updates 模式流式输出
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "旧金山天气怎么样？"}]},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        # chunk["data"] 是一个字典: {步骤名: 该步骤的状态更新}
        for step, data in chunk["data"].items():
            print(f"步骤: {step}")
            print(f"内容: {data['messages'][-1].content_blocks}")
```

**输出：**

```css
步骤: model
内容: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_xxx'}]

步骤: tools
内容: [{'type': 'text', 'text': '旧金山今天是晴天！'}]

步骤: model
内容: [{'type': 'text', 'text': '旧金山今天天气很好，是晴天！'}]
```

### 3.2 理解输出结构

每个 `chunk["data"]` 都是一个字典，键是步骤名，值是该步骤产生的状态更新：

```
第一个事件 (model):
  Agent 中的模型节点执行完毕
  → 模型决定调用 get_weather 工具
  → 输出包含 tool_call 类型的内容块

第二个事件 (tools):
  Agent 中的工具节点执行完毕
  → get_weather 工具执行完毕
  → 输出包含工具的返回结果

第三个事件 (model):
  模型节点再次执行（收到了工具结果）
  → 模型生成最终回答
  → 输出包含 text 类型的内容块
```

**这和第一节（Agent）中的 ReAct 循环完全对应：**

```
ReAct 循环:                    updates 流输出:
  模型思考 → 调用工具           → event: model (tool_call)
  工具执行                      → event: tools (result)
  模型根据结果回答              → event: model (text)
```

---

## 第四章：Token 级流式输出（messages 模式）

### 4.1 基本用法

`stream_mode="messages"` 把 LLM 生成的每一个 token 实时传出来——这就是"打字机效果"的实现原理

```python
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是晴天！"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "旧金山天气怎么样？"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]  # 每个 chunk 包含一个 token 和元数据
        print(f"节点: {metadata['langgraph_node']}")
        print(f"内容: {token.content_blocks}")
```

### 4.2 理解 token 级输出

**输出示例（简化）：**

```
节点: model
内容: [{'type': 'tool_call_chunk', 'name': 'get_weather', 'args': '', 'id': 'call_xxx'}]

节点: model
内容: [{'type': 'tool_call_chunk', 'args': '{"'}]

节点: model
内容: [{'type': 'tool_call_chunk', 'args': 'city'}]

节点: model
内容: [{'type': 'tool_call_chunk', 'args': '":"'}]

节点: model
内容: [{'type': 'tool_call_chunk', 'args': 'San Francisco'}]

节点: model
内容: [{'type': 'tool_call_chunk', 'args': '"}'}]

节点: tools
内容: [{'type': 'text', 'text': '旧金山今天是晴天！'}]

节点: model
内容: [{'type': 'text', 'text': '旧金山'}]

节点: model
内容: [{'type': 'text', 'text': '今天'}]

节点: model
内容: [{'type': 'text', 'text': '是'}]

节点: model
内容: [{'type': 'text', 'text': '晴天！'}]
```

**逐行解读：**

前面几行是模型在**逐步构造工具调用的 JSON 参数**——先传工具名和 ID，然后一点一点构造 `{"city":"San Francisco"}`。这就是第三节中 `tool_call_chunk` 的实际表现。中间一行是工具执行结果——这不是 LLM 生成的，而是工具节点直接输出的完整结果。最后几行是模型在**逐 token 生成最终回答**——"旧金山" → "今天" → "是" → "晴天！"

### 4.3 `chunk["data"]` 的结构

```python
token, metadata = chunk["data"]
```

| 字段 | 类型 | 内容 |
|------|------|------|
| `token` | `AIMessageChunk` 或其他消息类型 | 当前生成的 token（碎片） |
| `metadata` | `dict` | 元数据，包含来源节点等信息 |

**metadata 的重要字段：**

```python
metadata["langgraph_node"]  # 当前 token 来自哪个节点（"model" / "tools"）
metadata.get("lc_agent_name")  # Agent 的名称（多 Agent 时用于区分来源）
```

### 4.4 与第三节（Messages）的关联

还记得第三节中 `AIMessageChunk` 的概念吗？messages 模式的流式输出正是由一系列 `AIMessageChunk` 组成的。你可以用 `+` 运算符把它们拼接成完整的 `AIMessage`：

```python
from langchain.messages import AIMessageChunk

full_message = None

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "你好"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            # 逐字打印（打字机效果）
            if token.text:
                print(token.text, end="")

            # 累加碎片 → 完整消息
            full_message = token if full_message is None else full_message + token

            # 检查是否是最后一个碎片
            if token.chunk_position == "last":
                # full_message 现在是完整的消息
                if full_message.tool_calls:
                    print(f"\n工具调用: {full_message.tool_calls}")
                full_message = None  # 重置，准备接收下一条消息
```

**`chunk_position == "last"` 的作用：** 在流式输出中，模型可能生成多条消息（比如先是一条带工具调用的消息，然后是最终回答）。`chunk_position == "last"` 标记的是当前这条消息的最后一个碎片——此时你的累加结果就是一条完整的消息

---

## 第五章：自定义流式更新（custom 模式）

### 5.1 为什么需要自定义流？

`updates` 模式只能在步骤完成后通知你，`messages` 模式只传递 LLM 的 token。但有时候你需要在**工具执行过程中**发送进度信息：

```
"正在连接数据库..."       ← 工具执行中
"查询到 1000 条记录"       ← 工具执行中
"正在分析数据..."          ← 工具执行中
"分析完成！"               ← 工具执行完毕
```

这种中间状态的实时传递就需要 custom 模式。

### 5.2 使用 `get_stream_writer`

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """获取城市天气。"""
    writer = get_stream_writer()  # 获取流式写入器

    # 在工具执行过程中发送自定义更新
    writer(f"正在查询 {city} 的天气数据...")
    # ... 执行耗时操作（如 API 调用）...
    writer(f"已获取 {city} 的天气数据")

    return f"{city} 今天是晴天！"


agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
)

# 使用 custom 模式接收自定义更新
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "旧金山天气怎么样？"}]},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])  # 打印自定义更新内容
```

**输出：**

```
正在查询 San Francisco 的天气数据...
已获取 San Francisco 的天气数据
```

### 5.3 `get_stream_writer` vs `runtime.stream_writer`

在第四节（Tools）中，我们学过通过 `runtime.stream_writer` 发送流式更新。这两种方式的关系是：

```
runtime.stream_writer
  → 从 ToolRuntime 中获取
  → 只能在用 @tool 装饰器定义的工具中使用
  → 需要函数签名中有 runtime: ToolRuntime 参数

get_stream_writer()
  → 从 langgraph.config 导入的独立函数
  → 可以在任何 LangGraph 执行上下文中使用
  → 不需要 ToolRuntime 参数
  → 更灵活，但必须在 LangGraph 执行上下文中调用
```

**两种方式等价，选择哪个取决于你的工具定义方式：**

```python
# 方式一：通过 ToolRuntime（第四节讲过的）
@tool
def my_tool(city: str, runtime: ToolRuntime) -> str:
    """我的工具。"""
    runtime.stream_writer(f"处理中: {city}")
    return "结果"

# 方式二：通过 get_stream_writer（本节新学的）
def my_tool(city: str) -> str:
    """我的工具。"""
    writer = get_stream_writer()
    writer(f"处理中: {city}")
    return "结果"
```

**注意：** 使用了 `get_stream_writer` 的工具**只能在 LangGraph 执行上下文中运行**。如果你在 LangGraph 之外直接调用这个函数（如 `my_tool("北京")`），会报错，因为此时没有流式上下文可以写入

### 5.4 `writer()` 可以发送任意数据

`writer()` 不限于字符串——你可以发送任何可序列化的数据：

```python
writer("简单字符串")
writer({"progress": 50, "total": 100})          # 字典
writer(["step1", "step2", "step3"])              # 列表
writer(some_ai_message)                          # 甚至是 AIMessage 对象
```

---

## 第六章：组合多种模式——全方位监控

### 6.1 同时使用多种流式模式

实际应用中，你往往需要同时看到多种信息。传入列表即可组合：

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """获取城市天气。"""
    writer = get_stream_writer()
    writer(f"正在查询 {city} 的数据...")
    writer(f"已获取 {city} 的数据")
    return f"{city} 今天是晴天！"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "旧金山天气怎么样？"}]},
    stream_mode=["updates", "custom"],  # ← 同时接收两种流
    version="v2",
):
    # 通过 chunk["type"] 区分不同的流
    print(f"流类型: {chunk['type']}")
    print(f"数据: {chunk['data']}")
    print()
```

**输出（简化）：**

```
流类型: updates
数据: {'model': {'messages': [AIMessage(tool_calls=[{name: 'get_weather', ...}])]}}

流类型: custom
数据: 正在查询 San Francisco 的数据...

流类型: custom
数据: 已获取 San Francisco 的数据

流类型: updates
数据: {'tools': {'messages': [ToolMessage(content="旧金山今天是晴天！")]}}

流类型: updates
数据: {'model': {'messages': [AIMessage(content="旧金山今天天气很好！")]}}
```

### 6.2 事件的时间顺序

注意上面的输出顺序——它反映了 Agent 的实际执行顺序：

```
时间 →

1. model 步骤完成 (updates)
   → 模型决定调用工具

2. 工具开始执行
   → writer("正在查询...") (custom)
   → writer("已获取...") (custom)

3. tools 步骤完成 (updates)
   → 工具执行结果

4. model 步骤完成 (updates)
   → 最终回答
```

**关键理解：** `updates` 事件在步骤**完成后**才发出，而 `custom` 事件在步骤**执行过程中**就能发出。这就是 custom 模式的价值——它填补了步骤执行期间的"信息空白"

---

## 第七章：流式输出思考/推理过程

### 7.1 什么是推理 token？

某些高级模型（如 Claude 的 extended thinking、OpenAI 的 o-series）在生成最终回答之前会先进行"内部推理"。这些推理 token 可以被流式输出，让用户看到模型的思考过程

```
普通模型:
  用户: "1+1=?"
  模型: "2"  ← 直接给答案

推理模型:
  用户: "1+1=?"
  [推理] "用户问的是简单加法..."
  [推理] "1+1 的结果是 2"
  模型: "答案是 2"  ← 经过推理后给答案
```

### 7.2 流式输出推理 token

使用 `stream_mode="messages"` 并过滤 `"reasoning"` 类型的内容块：

```python
from langchain.agents import create_agent
from langchain.messages import AIMessageChunk
from langchain_anthropic import ChatAnthropic


def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是晴天！"


# 启用推理功能的模型
model = ChatAnthropic(
    model_name="claude-sonnet-4-6",
    thinking={
        "type": "enabled",
        "budget_tokens": 5000  # 允许最多 5000 个推理 token
    },
)

agent = create_agent(model=model, tools=[get_weather])

for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "旧金山天气怎么样？"}]},
    stream_mode="messages",
):
    if not isinstance(token, AIMessageChunk):
        continue

    # 过滤推理内容块
    reasoning = [b for b in token.content_blocks if b["type"] == "reasoning"]
    # 过滤文本内容块
    text = [b for b in token.content_blocks if b["type"] == "text"]

    if reasoning:
        print(f"[思考] {reasoning[0]['reasoning']}", end="")
    if text:
        print(text[0]["text"], end="")
```

**输出：**

```
[思考] 用户问的是旧金山的天气。我有一个获取天气的工具
[思考] ，让我调用 get_weather 并传入 "San Francisco"。
旧金山的天气是：今天是晴天！
```

### 7.3 跨提供商的标准化

这里再次体现了第三节中 `content_blocks` 标准化的价值：

```
Anthropic 原始格式:
  {"type": "thinking", "thinking": "..."}
       ↓ content_blocks 自动转换
标准格式:
  {"type": "reasoning", "reasoning": "..."}

OpenAI 原始格式:
  {"type": "reasoning", "summary": [...]}
       ↓ content_blocks 自动转换
标准格式:
  {"type": "reasoning", "reasoning": "..."}
```

不管用哪个提供商，你的过滤代码都是一样的：`b["type"] == "reasoning"`

---

## 第八章：高级模式——流式工具调用与子 Agent

### 8.1 同时流式传输 token 和完成的工具调用

在实际应用中，你可能需要：

1. **实时看到**工具调用参数的逐步构造（用于 UI 动画效果）
2. **获取完整的**已解析工具调用（用于业务逻辑）

通过组合 `messages` 和 `updates` 模式可以同时实现：

```python
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage


def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是晴天！"


agent = create_agent("openai:gpt-5.2", tools=[get_weather])


for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "波士顿天气怎么样？"}]},
    stream_mode=["messages", "updates"],  # 同时监听两种
    version="v2",
):
    if chunk["type"] == "messages":
        # messages 模式：实时 token
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            if token.text:
                print(token.text, end="|")  # 打字机效果
            if token.tool_call_chunks:
                print(token.tool_call_chunks)  # 工具调用碎片

    elif chunk["type"] == "updates":
        # updates 模式：完成的步骤
        for source, update in chunk["data"].items():
            msg = update["messages"][-1]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # 完整的、已解析的工具调用
                print(f"完整工具调用: {msg.tool_calls}")
            if isinstance(msg, ToolMessage):
                # 工具执行结果
                print(f"工具结果: {msg.content_blocks}")
```

**输出解读：**

```
# messages 模式的碎片（实时）
[{'name': 'get_weather', 'args': '', 'id': 'call_xxx', ...}]    ← 工具名到达
[{'args': '{"'}]                                                  ← JSON 开始
[{'args': 'city'}]                                                ← 键名
[{'args': '":"'}]
[{'args': 'Boston'}]                                              ← 值
[{'args': '"}'}]                                                  ← JSON 结束

# updates 模式的完整结果（步骤完成后）
完整工具调用: [{'name': 'get_weather', 'args': {'city': 'Boston'}, ...}]  ← 已解析
工具结果: [{'type': 'text', 'text': 'Boston 今天是晴天！'}]
The|weather|in|Boston|is|sunny|!|                                 ← 最终回答的 token
```

**两种模式的分工：**

```
messages 模式:
  → 提供"正在进行中"的实时信息
  → 碎片化的、未解析的 token
  → 用于 UI 实时渲染

updates 模式:
  → 提供"已完成"的步骤信息
  → 完整的、已解析的消息
  → 用于业务逻辑处理
```

### 8.2 从子 Agent 流式输出

当你的 Agent 内部包含多个 LLM（如一个"主管"Agent 调用一个"天气"Agent），你需要区分 token 来自哪个 Agent。

**构建多 Agent 架构：**

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是晴天！"


# 子 Agent：天气专家
weather_agent = create_agent(
    model=init_chat_model("openai:gpt-5.2"),
    tools=[get_weather],
    name="weather_agent",  # ← 给子 Agent 命名
)


# 把子 Agent 包装成工具
def call_weather_agent(query: str) -> str:
    """查询天气 Agent。"""
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].text


# 主管 Agent
agent = create_agent(
    model=init_chat_model("openai:gpt-5.2"),
    tools=[call_weather_agent],
    name="supervisor",  # ← 给主管 Agent 命名
)
```

**流式输出时区分来源：**

```python
from langchain.messages import AIMessageChunk

current_agent = None  # 跟踪当前活跃的 Agent

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "波士顿天气怎么样？"}]},
    stream_mode=["messages", "updates"],
    subgraphs=True,   # ← 关键：启用子图流式输出
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]

        # 通过 metadata 中的 lc_agent_name 区分来源
        agent_name = metadata.get("lc_agent_name")
        if agent_name and agent_name != current_agent:
            print(f"\n🤖 {agent_name}: ")
            current_agent = agent_name

        if isinstance(token, AIMessageChunk) and token.text:
            print(token.text, end="|")
```

**输出：**

```
🤖 supervisor:
[工具调用碎片: call_weather_agent...]

🤖 weather_agent:
[工具调用碎片: get_weather...]
Boston|今天|是|晴天|！|

🤖 supervisor:
波士顿|今天|天气|很好|，|是|晴天|！|
```

**关键参数：**

```
name="weather_agent"
  → 给 Agent 命名
  → 名字会出现在流式输出的 metadata 中
  → 也会附加到该 Agent 生成的所有 AIMessage 上

subgraphs=True
  → 启用子图的流式输出
  → 没有这个参数，你只能看到主 Agent 的流
  → 有了它，子 Agent 内部的每个 token 也会被传出来
```

---

## 第九章：流式输出与人机协作（Human-in-the-loop）

### 9.1 概述

当 Agent 配置了人机协作中间件（Human-in-the-loop），在执行工具前会暂停等待人类审批。流式输出可以和这个机制配合使用——在 `updates` 流中收集中断（interrupt）事件，然后通过 `Command` 恢复执行。

### 9.2 基本模式

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是晴天！"


agent = create_agent(
    "openai:gpt-5.2",
    tools=[get_weather],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True}  # get_weather 需要人类审批
        ),
    ],
    checkpointer=InMemorySaver(),  # 中断恢复需要 checkpointer
)

config = {"configurable": {"thread_id": "some_id"}}

# 第一次流式调用——会在工具执行前中断
interrupts = []
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "查查波士顿和旧金山的天气"}]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source == "__interrupt__":
                # 收集中断事件
                interrupts.extend(update)
                print("需要审批的工具调用:")
                for req in update[0].value["action_requests"]:
                    print(f"  {req['description']}")
```

**输出：**

```
需要审批的工具调用:
  Tool: get_weather, Args: {'city': 'Boston'}
需要审批的工具调用:
  Tool: get_weather, Args: {'city': 'San Francisco'}
```

**处理中断并恢复：**

```python
# 构造审批决定
decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {
        "decisions": [
            {"type": "approve"}  # 批准所有工具调用
            for _ in interrupt.value["action_requests"]
        ]
    }

# 用 Command(resume=decisions) 恢复执行
for chunk in agent.stream(
    Command(resume=decisions),  # ← 恢复执行
    config=config,              # ← 同一个 thread_id
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk) and token.text:
            print(token.text, end="")
```

**完整流程：**

```
第一次 stream():
  用户输入 → 模型决定调用工具 → 中断（等待审批）
                                    ↓
                              收集 interrupt 对象
                                    ↓
                              人类审核（approve/edit/reject）
                                    ↓
第二次 stream(Command(resume=...)):
  恢复执行 → 工具执行 → 模型生成最终回答
```

---

## 第十章：禁用流式输出

### 10.1 为什么要禁用？

有些场景下你不希望某个模型进行流式输出：

- **多 Agent 系统**中，只想流式输出主 Agent 的回答，子 Agent 的内部处理不需要
- **混合模型**时，某些模型不支持流式输出
- **部署到 LangSmith** 时，不想把某些模型的输出流式传到客户端

### 10.2 配置方式

```python
from langchain_openai import ChatOpenAI

# 禁用流式输出
model = ChatOpenAI(
    model="gpt-4.1",
    streaming=False  # ← 禁用
)

# 如果模型不支持 streaming 参数，用这个替代
# disable_streaming=True  ← 所有 chat model 基类都支持这个
```

---

## 第十一章：概念串联——流式输出在 Agent 生态中的位置

### 11.1 知识关系图

```
第一节 Agent ──────────────────────────────
  agent.invoke()  → 同步调用，等待完成后返回
  agent.stream()  → 流式调用，逐步返回（本节重点）

第二节 Model ──────────────────────────────
  model.invoke()  → 返回完整的 AIMessage
  model.stream()  → 返回 AIMessageChunk 碎片流
  → 本节的 messages 模式就是把 model.stream() 的输出传到 Agent 层面

第三节 Messages ────────────────────────────
  AIMessageChunk  → 流式输出的基本单元
  content_blocks  → 区分 text / reasoning / tool_call_chunk
  + 运算符        → 把碎片拼成完整消息
  → 本节大量使用这些概念

第四节 Tools ──────────────────────────────
  runtime.stream_writer  → 工具中发送流式更新（本节的 custom 模式）
  get_stream_writer()    → 另一种获取写入器的方式

第五节 Short-term Memory ───────────────────
  checkpointer    → human-in-the-loop 流式需要它来保存中断状态
  thread_id       → 恢复中断时需要同一个 thread

第六节 Streaming ← 你在这里 ────────────────
  三种流式模式: updates / messages / custom
  v2 格式统一输出结构
  推理 token 的流式传输
  多 Agent 的流式区分
  human-in-the-loop 的流式集成
```

---

## 第十二章：速查手册

### 三种流式模式

| 模式 | `chunk["type"]` | `chunk["data"]` 内容 | 粒度 |
|------|----------------|---------------------|------|
| `"updates"` | `"updates"` | `{步骤名: {messages: [...]}}` | 步骤完成后 |
| `"messages"` | `"messages"` | `(token, metadata)` 元组 | 每个 token |
| `"custom"` | `"custom"` | 你通过 writer() 发送的任意数据 | 自定义 |

### 流式输出 API

| 方法 | 同步/异步 | 用途 |
|------|----------|------|
| `agent.stream(...)` | 同步 | 在 for 循环中逐 chunk 处理 |
| `agent.astream(...)` | 异步 | 在 async for 循环中使用 |

### 常用参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `stream_mode` | 流式模式 | `"updates"` / `["messages", "custom"]` |
| `version` | 输出格式版本 | `"v2"`（推荐） |
| `subgraphs` | 是否包含子图流 | `True` / `False` |

### v2 chunk 结构

```python
{
    "type": "updates" | "messages" | "custom",  # 流的类型
    "ns": [...],                                  # 命名空间（子图路径）
    "data": ...                                   # 实际数据（格式取决于 type）
}
```

### 发送自定义流式更新的两种方式

| 方式 | 来源 | 适用场景 |
|------|------|---------|
| `runtime.stream_writer(data)` | ToolRuntime | 用 `@tool` 定义的工具 |
| `get_stream_writer()(data)` | langgraph.config | 任何 LangGraph 执行上下文 |

### 与前几节知识的关系

```
第一节（Agent）
  → agent.stream() 是本节的入口方法
  → ReAct 循环的每一步对应 updates 模式的一个事件

第二节（Model）
  → model.stream() 产生 AIMessageChunk
  → messages 模式将这些碎片传递到 Agent 层面

第三节（Messages）
  → AIMessageChunk 是 token 级流的基本单元
  → content_blocks 用于区分 text / reasoning / tool_call_chunk
  → + 运算符把碎片拼成完整消息

第四节（Tools）
  → stream_writer 发送 custom 模式的自定义更新
  → 工具调用的实时构造过程通过 messages 模式可见

第五节（Short-term Memory）
  → human-in-the-loop 流式需要 checkpointer
  → 中断和恢复通过 thread_id 关联

本节（Streaming）← 你在这里
  → 三种模式覆盖不同粒度的实时信息
  → v2 格式统一输出结构
  → 支持推理 token、子 Agent、人机协作的流式输出

下一步建议学习：
  → Frontend Streaming（前端流式）——用 React 构建实时 UI
  → Human-in-the-loop（人机协作）——中断与恢复的完整机制
  → Multi-agent（多 Agent）——多 Agent 架构的设计
```
