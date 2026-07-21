# LangChain 短期记忆（Short-term Memory）深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第五节。第一节学了 Agent（整体架构），第二节学了 Model（大脑），第三节学了 Messages（消息/纸条），第四节学了 Tools（手和脚），本节学习 Short-term Memory——Agent 的"对话记忆"。如果说模型是大脑、工具是手脚、消息是语言，那短期记忆就是让 Agent 在一次对话中"不健忘"的能力

---

## 第一章：为什么需要记忆？从"失忆症"说起

### 1.1 模型的先天缺陷——无状态

在第三节（Messages）中我们已经揭示了一个关键事实：**模型本身是无状态的**。每次调用 `model.invoke()` 都是一次全新的、独立的交互——模型不记得你上一次问了什么

```python
# 第一轮：告诉模型你的名字
response1 = model.invoke("我叫小明")
# → "你好，小明！"

# 第二轮：直接问名字——模型已经"失忆"
response2 = model.invoke("我叫什么名字？")
# → "抱歉，我不知道你的名字。"  ← 失忆了！
```

在第三节中，我们学到的解决方案是**手动维护消息列表**——每次调用时把完整的对话历史发给模型。这能工作，但你需要自己处理很多事情：保存历史、恢复历史、控制历史长度……

**短期记忆（Short-term Memory）就是 LangChain 对这套手动管理的自动化封装。** 它帮你：

1. **自动保存**每一轮对话的消息
2. **自动恢复**之前的对话状态
3. **自动管理**历史长度（防止超出上下文窗口）

### 1.2 短期记忆 vs 长期记忆

在开始之前，先区分两个概念，避免混淆：

| 维度 | 短期记忆（Short-term Memory） | 长期记忆（Long-term Memory） |
|:----:|:----------------------------:|:----------------------------:|
| **范围** | 单个对话（thread）内 | 跨对话、跨会话 |
| **内容** | 消息历史 + 自定义状态 | 用户偏好、知识库 |
| **生命周期** | 对话持续期间 | 永久（或显式删除前） |
| **实现方式** | Checkpointer | Store（第四节讲过） |
| **类比** | 和朋友聊天时记得前面说的话 | 记得这个朋友喜欢什么 |

### 1.3 Thread（线程/会话）的概念

LangChain 用 **thread（线程）** 来组织一次完整的对话，类似邮件中的"会话"——同一个话题下的多条来回消息被归为一组

```css
Thread "1"（和用户 A 的客服对话）:
  ├── Human: "我想退货"
  ├── AI: "请提供订单号"
  ├── Human: "订单 #12345"
  └── AI: "已为您办理退货"

Thread "2"（和用户 B 的技术支持对话）:
  ├── Human: "怎么重置密码"
  └── AI: "请访问设置页面..."
```

每个 thread 有独立的记忆——thread 1 的对话历史不会影响 thread 2。通过 `thread_id` 来区分不同的 thread

---

## 第二章：启用短期记忆——Checkpointer

### 2.1 最简实现

要让 Agent 拥有短期记忆，只需要做两件事：

1. 指定一个 **checkpointer**（检查点存储器）
2. 在调用时传入 **thread_id**

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 创建一个内存中的 checkpointer
# 它会把对话状态保存在 Python 进程的内存中
agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),  # ← 关键：指定 checkpointer
)

# 调用时必须传入 thread_id，告诉 Agent "这属于哪个对话"
agent.invoke(
    {"messages": [{"role": "user", "content": "你好！我叫小明。"}]},
    {"configurable": {"thread_id": "1"}},  # ← 关键：指定 thread_id
)
```

**这三行关键代码的作用：**

```css
InMemorySaver()
  → 创建一个"记忆存储器"
  → 每当 Agent 完成一步操作，状态就被自动保存到这里

checkpointer=InMemorySaver()
  → 把存储器"安装"到 Agent 上
  → Agent 现在会在每一步自动保存/读取状态

{"configurable": {"thread_id": "1"}}
  → 告诉 Agent 这是 thread "1" 的对话
  → Agent 会从存储器中找到 thread "1" 的历史状态
  → 如果是新 thread，就创建一个全新的状态
```

**有了 checkpointer 后的对话效果：**

```python
config = {"configurable": {"thread_id": "1"}}

# 第一轮：告诉 Agent 名字
agent.invoke(
    {"messages": [{"role": "user", "content": "你好！我叫小明。"}]},
    config
)
# → "你好，小明！有什么可以帮你的？"

# 第二轮：Agent 记得之前说过的话
agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config  # ← 同一个 thread_id
)
# → "你叫小明呀！你之前告诉我的。"  ← 记住了！
```

### 2.2 Checkpointer 的工作原理

```css
第一轮调用 (thread_id="1"):
  ┌──────────────────────────────────────────────┐
  │ 1. Agent 检查 checkpointer: thread "1" 存在吗？│
  │    → 不存在，创建空状态                         │
  │                                                │
  │ 2. 执行 Agent 逻辑                             │
  │    → 收到 HumanMessage("你好！我叫小明。")      │
  │    → 模型生成 AIMessage("你好，小明！...")       │
  │                                                │
  │ 3. 自动保存状态到 checkpointer                  │
  │    → 状态 = {messages: [Human("..."), AI("...")]}│
  └──────────────────────────────────────────────┘

第二轮调用 (thread_id="1"):
  ┌──────────────────────────────────────────────┐
  │ 1. Agent 检查 checkpointer: thread "1" 存在吗？│
  │    → 存在！加载之前保存的状态                    │
  │    → 状态 = {messages: [Human("..."), AI("...")]}│
  │                                                │
  │ 2. 把新消息追加到已有的消息列表中                 │
  │    → [Human("你好！"), AI("你好小明！"),          │
  │       Human("我叫什么名字？")]                   │
  │                                                │
  │ 3. 模型收到完整的消息列表，生成回答               │
  │    → "你叫小明呀！"                              │
  │                                                │
  │ 4. 自动保存更新后的状态                          │
  └──────────────────────────────────────────────┘
```

**关键理解：** Checkpointer 的保存不仅仅发生在对话结束时。**每完成一步操作**（比如模型生成了回复、工具执行完毕），状态都会被保存一次。这意味着即使程序中途崩溃，也能从最近的"检查点"恢复

### 2.3 生产环境的 Checkpointer

`InMemorySaver` 只适合开发测试——程序一重启，所有对话历史就丢失了。生产环境需要用持久化的 checkpointer：

```bash
# 安装 PostgreSQL checkpointer
pip install langgraph-checkpoint-postgres
```

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动在 PostgreSQL 中创建所需的表

    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,  # 状态保存到数据库
    )
```

**不同 Checkpointer 的对比：**

| Checkpointer | 持久性 | 适用场景 |
|:------------:|:-----:|:-------:|
| `InMemorySaver` | 程序重启即丢失 | 开发测试 |
| `PostgresSaver` | 持久化到 PostgreSQL | 生产环境（推荐） |
| `SqliteSaver` | 持久化到 SQLite 文件 | 轻量级生产、单机部署 |
| Azure Cosmos DB | 持久化到云端 | Azure 云环境 |

---

## 第三章：自定义 Agent 状态——不只是消息

### 3.1 默认状态：AgentState

默认情况下，Agent 的状态只有一个 `messages` 字段——就是对话历史。这由 `AgentState` 定义：

```python
# AgentState 的内部定义（简化版）
class AgentState:
    messages: list[BaseMessage]  # 对话历史
```

但很多场景下，你需要在状态中存储更多信息——用户 ID、偏好设置、中间计算结果等

### 3.2 扩展状态

通过继承 `AgentState` 来添加自定义字段：

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):
    user_id: str          # 自定义字段：用户 ID
    preferences: dict     # 自定义字段：用户偏好


agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,  # ← 使用自定义状态
    checkpointer=InMemorySaver(),
)

# 调用时可以传入自定义字段的值
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "你好"}],
        "user_id": "user_123",              # ← 自定义字段
        "preferences": {"theme": "dark"}    # ← 自定义字段
    },
    {"configurable": {"thread_id": "1"}}
)
```

**自定义状态的数据流：**

```
调用时传入:
  messages: [...], user_id: "user_123", preferences: {theme: "dark"}
       │
       ▼
Agent 状态 (CustomAgentState):
  ├── messages: [HumanMessage("你好")]       ← 自动管理
  ├── user_id: "user_123"                    ← 你传入的
  └── preferences: {theme: "dark"}           ← 你传入的
       │
       ▼
工具可以通过 runtime.state 访问:
  runtime.state["user_id"]       → "user_123"
  runtime.state["preferences"]   → {theme: "dark"}
       │
       ▼
Checkpointer 自动保存整个状态（包括自定义字段）
  → 下次调用同一个 thread 时，所有字段都会被恢复
```

**这和第四节（Tools）中 `ToolRuntime.state` 的关联：** 自定义状态中的字段可以被工具通过 `runtime.state` 访问。这就是为什么在第四节中工具能读到 `runtime.state["user_preferences"]`——那个字段就是在自定义状态中定义的

---

## 第四章：对话历史管理——当记忆太多时

### 4.1 问题：记忆不是越多越好

随着对话进行，消息列表不断增长，这带来三个实际问题：

```css
问题一：超出上下文窗口
  模型能处理的 token 有上限（比如 128K）
  50 轮对话后，消息列表可能有 10 万 token
  → 模型直接报错："超出最大长度"

问题二：性能下降
  即使没超出窗口，模型在处理长上下文时也会变慢
  → 响应时间从 2 秒变成 15 秒

问题三：成本飙升
  API 按 token 计费
  每轮对话都发送完整历史
  → 第 50 轮对话的成本是第 1 轮的 50 倍
```

**解决思路：** 我们需要某种方式来"遗忘"不再重要的旧信息，同时保留关键上下文。LangChain 提供了三种主要策略：

```
策略一：裁剪消息（Trim）
  → 保留最近的 N 条消息，丢弃更早的
  → 简单直接，但可能丢失重要的早期信息

策略二：删除消息（Delete）
  → 从状态中永久删除特定消息
  → 更精细的控制，但需要你决定删什么

策略三：摘要消息（Summarize）
  → 用模型把旧消息压缩成一段摘要
  → 保留了信息要点，但增加了模型调用成本
```

### 4.2 Middleware（中间件）——消息管理的执行者

在讲具体策略之前，需要先理解 **Middleware（中间件）** 的概念。中间件是插入在 Agent 工作流中的"拦截器"——它可以在模型被调用**之前**或**之后**执行自定义逻辑。==消息的裁剪、删除、摘要都通过中间件来实现==

```
Agent 工作流（有中间件）:

  用户输入
    │
    ▼
  ┌─────────────┐
  │ before_model │ ← 模型调用前的中间件（如：裁剪消息）
  │  中间件       │
  └─────────────┘
    │
    ▼
  ┌─────────────┐
  │   模型       │ ← 收到的是处理后的消息
  └─────────────┘
    │
    ▼
  ┌─────────────┐
  │ after_model  │ ← 模型调用后的中间件（如：验证输出）
  │  中间件       │
  └─────────────┘
    │
    ├── 有工具调用 → 执行工具 → 回到 before_model
    │
    └── 无工具调用 → 返回最终回答
```

LangChain 提供了两个中间件装饰器：

| 装饰器 | 执行时机 | 典型用途 |
|--------|---------|---------|
| `@before_model` | 模型被调用之前 | 裁剪消息、注入上下文、修改提示词 |
| `@after_model` | 模型返回之后 | 验证输出、删除敏感消息、记录日志 |

### 4.3 策略一：裁剪消息（Trim Messages）

裁剪是最简单的策略——保留最近的几条消息，丢弃更早的。

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """保留第一条消息（通常是 SystemMessage）和最近几条消息。"""
    messages = state["messages"]

    # 如果消息总数不多，不需要裁剪
    if len(messages) <= 3:
        return None  # 返回 None 表示"不做任何修改"

    # 保留第一条消息（通常是 SystemMessage 或初始指令）
    first_msg = messages[0]

    # 保留最近几条消息
    # 根据消息数量的奇偶性选择保留数量，确保消息列表以 Human 开头
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]

    # 拼接：第一条 + 最近几条
    new_messages = [first_msg] + recent_messages

    # 返回更新后的消息列表
    # RemoveMessage(id=REMOVE_ALL_MESSAGES) 先清空所有消息
    # 然后用 *new_messages 重新填充
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 先清空
            *new_messages                           # 再填入保留的消息
            """
            *用于解包可迭代对象
			比如：
				列表 list
				元组 tuple
				字符串 str

			**用于解包字典 dict
            """
        ]
    }


# 创建带裁剪中间件的 Agent
agent = create_agent(
    your_model_here,
    tools=your_tools_here,
    middleware=[trim_messages],     # ← 注册中间件
    checkpointer=InMemorySaver(),
)
```

**`RemoveMessage` 和 `REMOVE_ALL_MESSAGES` 详解：**

`RemoveMessage` 是 LangChain 提供的一种特殊"指令消息"——它不是真正的对话消息，而是告诉状态管理系统"删除某条消息"

```python
# 删除特定消息（通过 id 指定）
RemoveMessage(id=message.id)      # 删除 id 对应的那一条消息

# 删除所有消息
RemoveMessage(id=REMOVE_ALL_MESSAGES)  # 清空整个消息列表
```

**裁剪后的实际效果：**

```
裁剪前（10 条消息）:
  [0] SystemMessage("你是助手")
  [1] HumanMessage("你好")
  [2] AIMessage("你好！")
  [3] HumanMessage("天气怎么样")
  [4] AIMessage("让我查一下...")
  [5] ToolMessage("晴天 25°C")
  [6] AIMessage("今天是晴天")
  [7] HumanMessage("推荐个餐厅")
  [8] AIMessage("推荐火锅...")
  [9] HumanMessage("我叫什么名字？")    ← 最新消息

裁剪后（保留第一条 + 最近 4 条）:
  [0] SystemMessage("你是助手")          ← 第一条（保留）
  [6] AIMessage("今天是晴天")            ← 最近的
  [7] HumanMessage("推荐个餐厅")
  [8] AIMessage("推荐火锅...")
  [9] HumanMessage("我叫什么名字？")

→ 模型只看到这 5 条消息
→ 它不知道用户之前说过"你好"
→ 它也不知道用户叫什么名字了（名字在被裁掉的早期消息中）
```

**裁剪的局限性：** 简单粗暴地丢弃旧消息可能会丢失重要信息（如用户的名字、之前的约定）。这就是为什么还有"摘要"策略

### 4.4 策略二：删除消息（Delete Messages）

删除是比裁剪更精细的控制——你可以选择性地删除特定的消息，而不是简单地按时间截断

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """在模型回答后，删除最早的两条消息来保持历史可控。"""
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {
            "messages": [
                RemoveMessage(id=m.id) for m in messages[:2]
            ]
        }
    return None
```

**注意这里用的是 `@after_model` 而不是 `@before_model`：**

```
@before_model 裁剪:
  → 在模型"看到"消息之前就裁剪了
  → 模型看到的是已裁剪的消息列表
  → 不影响状态中保存的实际消息（只影响模型的输入）

@after_model 删除:
  → 模型已经看到了完整的消息列表并回答了
  → 删除操作影响的是状态中保存的实际消息
  → 下次调用时，被删除的消息就真的不存在了
```

**删除消息的注意事项：**

删除消息后，剩余的消息列表必须仍然是"合法的"。不同模型提供商有不同的要求：

```
常见要求：
  1. 消息列表可能需要以 user 消息开头（某些提供商）
  2. 带 tool_calls 的 AIMessage 后面必须跟对应的 ToolMessage
     → 不能只删 ToolMessage 而留下 AIMessage（模型会困惑）
     → 也不能只删 AIMessage 而留下 ToolMessage（缺少关联）

安全的做法：
  → 成对删除：如果删 HumanMessage，也删对应的 AIMessage
  → 工具链完整删除：AIMessage(tool_calls) + ToolMessage 一起删
```

### 4.5 策略三：摘要消息（Summarize Messages）

摘要是最"智能"的策略——用模型把旧消息压缩成一段摘要，既控制了长度，又保留了关键信息。

```
裁剪策略:
  [消息1] [消息2] [消息3] [消息4] [消息5] [消息6]
         ↓ 裁剪前4条
  [消息5] [消息6]
  → 信息丢失！

摘要策略:
  [消息1] [消息2] [消息3] [消息4] [消息5] [消息6]
         ↓ 把前4条消息让模型总结
  [摘要："用户叫小明，询问了天气和餐厅"] [消息5] [消息6]
  → 信息被压缩保留！
```

LangChain 提供了内置的 `SummarizationMiddleware`：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",     # 用更便宜的模型来做摘要
            trigger=("tokens", 4000), # 当消息总 token 超过 4000 时触发摘要
            keep=("messages", 20)     # 保留最近的 20 条消息不被摘要
        )
    ],
    checkpointer=InMemorySaver(),
)
```

**三个参数详解：**

```
model="gpt-4.1-mini"
  → 用哪个模型来生成摘要
  → 建议用比主模型更便宜/快速的模型（反正只是做摘要）
  → 省成本、省时间

trigger=("tokens", 4000)
  → 触发摘要的条件
  → 当消息列表的总 token 数超过 4000 时，自动触发摘要
  → 4000 以下不触发（消息不多，不需要压缩）

keep=("messages", 20)
  → 保留最近的 20 条消息不被摘要
  → 只有前面更早的消息会被压缩成摘要
  → 这样最近的对话仍然是完整的
```

**摘要的执行过程：**

```
对话已有 50 条消息，总 token = 5000（超过 4000）

触发摘要:
  ├── 保留最近 20 条消息（不动）
  ├── 把前 30 条消息发给 gpt-4.1-mini
  │   → "用户叫小明，是一名程序员。之前讨论了天气查询功能的实现，
  │      用户希望支持多城市查询。还聊了关于 Python 的装饰器用法。"
  └── 用摘要替换前 30 条消息

处理后:
  [摘要] + [最近 20 条消息]
  → 总 token 大幅减少
  → 关键信息被保留在摘要中
```

**实际使用效果：**

```python
config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "你好，我叫小明"}, config)
agent.invoke({"messages": "写一首关于猫的短诗"}, config)
agent.invoke({"messages": "再写一首关于狗的"}, config)
# ... 很多轮对话后 ...
final = agent.invoke({"messages": "我叫什么名字？"}, config)
print(final["messages"][-1].text)
# → "你叫小明！"  ← 即使早期消息被摘要了，名字仍然被保留
```

### 4.6 三种策略对比

| 维度 | 裁剪（Trim） | 删除（Delete） | 摘要（Summarize） |
|------|-------------|---------------|-------------------|
| **信息保留** | 低——旧信息直接丢弃 | 中——可选择性保留 | 高——压缩但保留要点 |
| **实现复杂度** | 低 | 中 | 高 |
| **额外成本** | 无 | 无 | 需要额外的模型调用 |
| **延迟影响** | 无 | 无 | 摘要生成需要时间 |
| **适用场景** | 对历史不敏感的任务 | 需要精细控制的场景 | 需要保留上下文的长对话 |

---

## 第五章：访问和修改记忆的多种方式

短期记忆不仅可以在对话中自动管理，你还可以从多个"入口"主动读写它

### 5.1 入口一：在工具中访问记忆

这在第四节（Tools）中已经详细讲过。回顾一下：

#### 读取状态

```python
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str  # 自定义状态字段


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """查询用户信息。"""
    # 通过 runtime.state 读取状态
    user_id = runtime.state["user_id"]
    return "用户是小明" if user_id == "user_123" else "未知用户"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "查询用户信息",
    "user_id": "user_123"
})
# → "用户是小明"
```

#### 写入状态

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):
    user_name: str   # 将被工具更新的字段


class CustomContext(BaseModel):
    user_id: str


@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState]) -> Command:
    """查询并更新用户信息。"""
    user_id = runtime.context.user_id
    name = "小明" if user_id == "user_123" else "未知"

    # 返回 Command 来更新状态
    return Command(update={
        "user_name": name,                          # 更新自定义字段
        "messages": [
            ToolMessage(
                "用户信息已更新",
                tool_call_id=runtime.tool_call_id   # 匹配工具调用 ID
            )
        ]
    })
```

**`ToolRuntime[CustomContext, CustomState]` 的泛型语法解释：**

这里方括号中有两个类型参数：
- 第一个 `CustomContext` 是 context 的类型
- 第二个 `CustomState` 是 state 的类型

这让 IDE 能正确提示 `runtime.context.user_id` 和 `runtime.state["user_name"]`

### 5.2 入口二：动态提示词（Dynamic Prompt）

通过中间件动态生成系统提示词，让提示词能访问上下文信息：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict


class CustomContext(TypedDict):
    user_name: str


def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是大晴天！"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """根据上下文动态生成系统提示词。"""
    # 从运行时上下文中获取用户名
    user_name = request.runtime.context["user_name"]
    return f"你是一个有帮助的助手。请称呼用户为 {user_name}。"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],    # ← 注册动态提示词中间件
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山天气怎么样？"}]},
    context=CustomContext(user_name="小明"),
)
```

**输出效果：**

```
Human: 旧金山天气怎么样？
AI: [调用 get_weather(city="San Francisco")]
Tool: 旧金山今天是大晴天！
AI: 小明你好，旧金山今天是大晴天！  ← 使用了动态注入的用户名
```

**`@dynamic_prompt` 的工作原理：**

```
普通的 system_prompt:
  create_agent(..., system_prompt="你是助手")
  → 每次对话都用同样的提示词
  → 无法包含动态信息

@dynamic_prompt:
  → 每次模型被调用前，执行你的函数
  → 函数可以访问 runtime（上下文、状态等）
  → 返回的字符串作为系统提示词
  → 每次调用都可以生成不同的提示词
```

### 5.3 入口三：`@before_model` 中间件

在模型调用前执行，可以修改消息列表、注入额外信息、裁剪历史等

```
执行流程:
  START → before_model → 模型 → (工具?) → before_model → 模型 → ... → END
              ↑                                  ↑
         每次模型调用前都执行              工具执行后再次经过
```

前面的裁剪消息示例就是 `@before_model` 的典型用法

### 5.4 入口四：`@after_model` 中间件

在模型返回后执行，可以验证输出、删除敏感信息、记录日志等

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """检查模型回复中是否包含敏感词，如果有则删除该回复。"""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]  # 模型刚生成的回复

    # 检查是否包含敏感词
    if any(word in last_message.content for word in STOP_WORDS):
        # 删除这条包含敏感词的消息
        return {"messages": [RemoveMessage(id=last_message.id)]}

    # 不包含敏感词，不做修改
    return None


agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)
```

**`@before_model` vs `@after_model` 的执行位置对比：**

```
@before_model 的执行位置:
  START → [before_model] → 模型 → 工具 → [before_model] → 模型 → END
            ↑ 这里                          ↑ 这里

@after_model 的执行位置:
  START → 模型 → [after_model] → 工具 → 模型 → [after_model] → END
                   ↑ 这里                         ↑ 这里
```

### 5.5 四种入口总结

| 入口 | 执行时机 | 能做什么 | 典型用途 |
|------|---------|---------|---------|
| 工具中（ToolRuntime） | 工具被调用时 | 读写状态和上下文 | 查询用户信息、更新偏好 |
| `@dynamic_prompt` | 模型调用前 | 动态生成系统提示词 | 个性化提示、多语言 |
| `@before_model` | 模型调用前 | 修改消息列表和状态 | 裁剪消息、注入上下文 |
| `@after_model` | 模型返回后 | 检查/修改模型输出 | 内容过滤、日志记录 |

---

## 第六章：概念串联——短期记忆在 Agent 生态中的位置

### 6.1 把前五节串在一起

```
第一节 Agent ──────────────────────────────
  create_agent() 的完整参数：
    model       → 第二节
    tools       → 第四节
    checkpointer → 本节（短期记忆的持久化）
    state_schema → 本节（自定义状态）
    middleware   → 本节（消息管理策略）

第二节 Model ──────────────────────────────
  模型是无状态的
  → 需要每次都接收完整的消息列表
  → 短期记忆帮你自动管理这个列表

第三节 Messages ────────────────────────────
  消息列表随对话增长
  → 可能超出上下文窗口
  → 短期记忆的裁剪/摘要策略解决这个问题
  → RemoveMessage 是操作消息列表的工具

第四节 Tools ──────────────────────────────
  工具通过 ToolRuntime.state 读取短期记忆
  工具通过 Command 写入短期记忆
  → 工具和记忆系统是紧密协作的

第五节 Short-term Memory ← 你在这里 ────────
  Checkpointer 提供持久化的对话状态存储
  AgentState 定义状态的结构
  Middleware 在模型调用前后处理消息
  三种策略管理消息历史长度
```

### 6.2 完整的数据流追踪

```
用户发送消息 "我叫什么名字？" (thread_id="1")
     │
     ▼
Checkpointer 加载 thread "1" 的状态
  → messages: [之前的 30 条消息...]
  → user_id: "user_123"
     │
     ▼
@before_model 中间件执行
  → trim_messages 把 30 条裁剪到 5 条
  → 或 SummarizationMiddleware 把旧消息压缩成摘要
     │
     ▼
模型收到处理后的消息列表
  → [SystemMessage, 摘要, 最近几条消息, HumanMessage("我叫什么名字？")]
  → 模型生成回答（可能调用工具）
     │
     ▼
如果有工具调用:
  → 工具通过 runtime.state 访问状态
  → 工具可能通过 Command 更新状态
  → 工具结果变成 ToolMessage
  → 回到 @before_model，再次调用模型
     │
     ▼
@after_model 中间件执行
  → 验证输出、删除敏感信息等
     │
     ▼
Checkpointer 保存更新后的状态
  → 包括新增的 HumanMessage 和 AIMessage
  → 包括工具可能更新的自定义字段
     │
     ▼
返回最终回答给用户
```

---

## 第七章：速查手册

### Checkpointer 选择

| 类型 | 适用场景 | 持久性 |
|------|---------|-------|
| `InMemorySaver` | 开发测试 | 程序重启即丢失 |
| `PostgresSaver` | 生产环境 | 永久（数据库） |
| `SqliteSaver` | 轻量部署 | 永久（文件） |

### 消息管理策略

| 策略 | 实现方式 | 信息保留 | 额外成本 |
|------|---------|---------|---------|
| 裁剪 | `@before_model` + `RemoveMessage` | 低 | 无 |
| 删除 | `@after_model` + `RemoveMessage` | 中 | 无 |
| 摘要 | `SummarizationMiddleware` | 高 | 需要模型调用 |

### 中间件装饰器

| 装饰器 | 时机 | 参数 | 返回值 |
|--------|------|------|--------|
| `@before_model` | 模型调用前 | `state`, `runtime` | `dict` 或 `None` |
| `@after_model` | 模型返回后 | `state`, `runtime` | `dict` 或 `None` |
| `@dynamic_prompt` | 模型调用前 | `request: ModelRequest` | `str`（提示词） |

### RemoveMessage 用法

| 操作 | 代码 |
|------|------|
| 删除特定消息 | `RemoveMessage(id=message.id)` |
| 删除所有消息 | `RemoveMessage(id=REMOVE_ALL_MESSAGES)` |
| 批量删除 | `[RemoveMessage(id=m.id) for m in messages[:N]]` |

### 与前几节知识的关系

```
第一节（Agent）
  → Agent 的 checkpointer 参数启用短期记忆
  → Agent 的 state_schema 参数自定义记忆结构
  → Agent 的 middleware 参数控制记忆管理策略

第二节（Model）
  → 模型无状态 → 需要短期记忆来维持对话连贯性
  → 模型有上下文窗口限制 → 需要裁剪/摘要策略

第三节（Messages）
  → 消息列表是短期记忆的核心内容
  → RemoveMessage 操作消息列表
  → 消息类型决定了哪些可以安全删除

第四节（Tools）
  → 工具通过 ToolRuntime.state 读取记忆
  → 工具通过 Command 写入记忆
  → 工具是记忆系统的主要"消费者"

本节（Short-term Memory）← 你在这里
  → Checkpointer 持久化对话状态
  → AgentState 定义状态结构
  → Middleware 管理消息历史
  → 三种策略应对上下文窗口限制

下一步建议学习：
  → Long-term Memory（长期记忆）——跨会话的持久化数据
  → Middleware（中间件）——更多中间件的高级用法
  → Streaming（流式输出）——实时传输 Agent 的执行过程
```
