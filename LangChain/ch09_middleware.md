# LangChain 中间件（Middleware）深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第八节。前七节分别学了 Agent（整体架构）、Model（大脑）、Messages（消息）、Tools（手和脚）、Short-term Memory（对话记忆）、Streaming（实时播报）、Structured Output（结构化输出）。本节学习 Middleware——Agent 的**"拦截器"和"增强器"**。如果说前面七节构建了 Agent 的核心能力，那中间件解决的是**"怎么在 Agent 运行的每个环节插入额外的控制逻辑"**的问题

---

## 第一章：为什么需要中间件？

### 1.1 没有中间件的困境

假设你有一个能调用工具的 Agent，它已经能正常工作了。但随着投入生产，你发现：

- Agent 有时候进入死循环，疯狂调用同一个工具 → **需要限制调用次数**
- 对话太长，上下文窗口爆了 → **需要自动压缩历史**
- 某些工具涉及资金操作，不能让 Agent 自动执行 → **需要人工审批**
- 用户发来的消息包含身份证号、信用卡号 → **需要敏感信息检测**
- 主模型 API 偶尔宕机 → **需要自动切换到备用模型**
- Agent 有 50 个工具，每次都全部传给模型很浪费 token → **需要智能筛选工具**

这些需求有一个共同特点：它们不是 Agent 的"核心逻辑"（思考→调用工具→回答），而是**围绕核心逻辑的控制、监控、增强**。如果把这些逻辑直接写进 Agent 代码里，代码会变得极其复杂且难以维护

### 1.2 中间件的解决思路

中间件的核心思想是：**在 Agent 执行的关键节点插入"钩子"（hook），在不修改核心逻辑的情况下添加额外功能。**

```
没有中间件:
  用户输入 → 模型思考 → 调用工具 → 模型回答

有中间件（概念示意）:
  用户输入
    → [中间件A: 检测敏感信息] 
    → 模型思考
    → [中间件B: 限制调用次数]
    → 调用工具
    → [中间件C: 重试失败的调用]
    → 模型回答
    → [中间件D: 压缩历史]
```

**中间件的两大价值：**

1. **解耦**——每个中间件只负责一件事，互不干扰，可以自由组合
2. **复用**——写好的中间件可以在不同 Agent 之间共享，LangChain 也提供了大量开箱即用的预置中间件

### 1.3 中间件在 Agent 循环中的位置

回忆第一节学的 Agent 核心循环：

```
      ┌──────────────────────────────────┐
      │         Agent 核心循环            │
      │                                   │
      │    调用模型 ──→ 模型返回结果       │
      │       ↑              │            │
      │       │      有工具调用？          │
      │       │        ↙      ↘           │
      │       │      是        否          │
      │       │       ↓        ↓          │
      │    执行工具      返回最终结果      │
      └──────────────────────────────────┘
```

中间件在这个循环的**每个关键步骤前后**都有钩子：

```
      ┌──────────────────────────────────────────────┐
      │         Agent 循环 + 中间件钩子               │
      │                                               │
      │    [模型调用前钩子] ← PII检测、工具筛选等      │
      │         ↓                                     │
      │    调用模型                                    │
      │         ↓                                     │
      │    [模型调用后钩子] ← 模型重试、备用模型等      │
      │         ↓                                     │
      │    有工具调用？                                │
      │      ↙      ↘                                │
      │    是        否 → 返回结果                     │
      │     ↓                                         │
      │    [工具执行前钩子] ← 人工审批、调用限制等      │
      │         ↓                                     │
      │    执行工具                                    │
      │         ↓                                     │
      │    [工具执行后钩子] ← 工具重试、结果检查等      │
      │         ↓                                     │
      │    回到调用模型 ↑                              │
      └──────────────────────────────────────────────┘
```

---

## 第二章：中间件的基本用法

### 2.1 添加中间件

通过 `create_agent` 的 `middleware` 参数传入中间件列表：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[
        SummarizationMiddleware(...),   # 自动压缩长对话
        HumanInTheLoopMiddleware(...)   # 人工审批工具调用
    ],
)
```

**中间件的执行顺序：** 列表中的中间件按顺序执行。如果多个中间件影响同一个钩子（如都在"模型调用前"做处理），它们会按列表顺序依次执行

### 2.2 中间件分类总览

LangChain 提供的预置中间件可以按功能分为几大类：

```
┌─────────────────────────────────────────────────────────────┐
│                    中间件功能分类                             │
│                                                              │
│  📊 监控与安全                                               │
│    ├── PII Detection      敏感信息检测（邮箱/信用卡/IP等）   │
│    └── Model Call Limit   模型调用次数限制                   │
│                                                              │
│  🔄 容错与重试                                               │
│    ├── Model Fallback     模型故障自动切换                   │
│    ├── Model Retry        模型调用失败自动重试               │
│    └── Tool Retry         工具调用失败自动重试               │
│                                                              │
│  🧠 上下文管理                                               │
│    ├── Summarization      自动压缩长对话历史                 │
│    └── Context Editing    清理旧的工具输出                   │
│                                                              │
│  🎮 执行控制                                                 │
│    ├── Human-in-the-loop  人工审批工具调用                   │
│    ├── Tool Call Limit    工具调用次数限制                   │
│    └── LLM Tool Selector  智能工具筛选                      │
│                                                              │
│  🛠️ 能力扩展                                                │
│    ├── To-do List         任务规划与跟踪                     │
│    ├── Shell Tool         命令行执行                         │
│    ├── File Search        文件搜索（glob/grep）             │
│    ├── Filesystem         文件系统读写（短期+长期记忆）      │
│    ├── Subagent           子Agent委派任务                    │
│    └── LLM Tool Emulator  用LLM模拟工具（测试用）           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 第三章：上下文管理类中间件

### 3.1 SummarizationMiddleware——自动压缩长对话

**解决的问题：** 对话越来越长，快要超出模型的上下文窗口了

**工作原理：** 监控对话的 token 数量，当达到阈值时，自动用一个（通常更便宜的）模型把旧的对话内容压缩成摘要，只保留最近的几条消息

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",        # 用便宜的模型做摘要
            trigger=("tokens", 4000),     # 当 token 数达到 4000 时触发
            keep=("messages", 20),        # 保留最近 20 条消息不压缩
        ),
    ],
)
```

**执行流程：**

```
对话进行中...
  消息1 (200 tokens)
  消息2 (300 tokens)
  ...
  消息50 (总计 4500 tokens)  ← 超过 trigger 阈值 4000！

触发摘要:
  → 保留最近 20 条消息（keep）
  → 把前 30 条消息发给 gpt-4.1-mini 生成摘要
  → 用摘要替换前 30 条消息

压缩后:
  [摘要消息] (500 tokens)    ← 替代了前 30 条消息
  消息31-50 (2000 tokens)    ← 保留不变
  总计: 2500 tokens          ← 大幅缩减
```

**触发条件（trigger）的三种写法：**

```python
# 方式一：token 数量
trigger=("tokens", 4000)        # token 数 ≥ 4000 时触发

# 方式二：消息数量
trigger=("messages", 50)        # 消息数 ≥ 50 时触发

# 方式三：上下文窗口占比
trigger=("fraction", 0.8)       # 占模型上下文窗口的 80% 时触发

# 组合条件（OR 逻辑——满足任一即触发）
trigger=[("tokens", 3000), ("messages", 6)]
```

**保留策略（keep）的三种写法：**

```python
keep=("messages", 20)      # 保留最近 20 条消息
keep=("tokens", 2000)      # 保留最近约 2000 tokens 的消息
keep=("fraction", 0.3)     # 保留占上下文窗口 30% 的消息
```

**与第五节（Short-term Memory）的关系：** SummarizationMiddleware 解决的是 memory 的"溢出"问题。第五节学的是如何保存对话历史，本节的摘要中间件则是当历史太长时如何"智能裁剪"

### 3.2 ContextEditingMiddleware——清理旧工具输出

**解决的问题：** Agent 调用了很多工具，工具返回的长文本占用大量 token，但旧的工具结果已经不重要了

**与 Summarization 的区别：** Summarization 压缩整体对话，ContextEditing 专门针对工具输出——它把旧的 ToolMessage 内容替换为占位符（如 `[cleared]`），保留消息结构但释放 token 空间

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100000,   # token 数达到 10 万时触发
                    keep=3,           # 保留最近 3 次工具调用的结果
                ),
            ],
        ),
    ],
)
```

**清理前 vs 清理后：**

```
清理前:
  ToolMessage (search_tool): "这是一篇 5000 字的搜索结果..."  ← 旧的
  ToolMessage (search_tool): "这是另一篇 3000 字的结果..."    ← 旧的
  ToolMessage (database_tool): "查询结果: [100行数据]..."     ← 最近的，保留
  ToolMessage (search_tool): "最新搜索结果..."               ← 最近的，保留
  ToolMessage (database_tool): "最新查询结果..."             ← 最近的，保留

清理后:
  ToolMessage (search_tool): "[cleared]"                     ← 内容被替换
  ToolMessage (search_tool): "[cleared]"                     ← 内容被替换
  ToolMessage (database_tool): "查询结果: [100行数据]..."     ← 保留（最近3次之一）
  ToolMessage (search_tool): "最新搜索结果..."               ← 保留
  ToolMessage (database_tool): "最新查询结果..."             ← 保留
```

**ClearToolUsesEdit 的可选配置：**

```python
ClearToolUsesEdit(
    trigger=100000,            # 触发阈值（token 数）
    keep=3,                    # 保留最近 N 次工具结果
    clear_tool_inputs=False,   # 是否也清除工具调用的参数
    exclude_tools=["search"],  # 排除特定工具（这些工具的结果永远不清理）
    placeholder="[cleared]",   # 替换占位符文本
)
```

---

## 第四章：容错与重试类中间件

### 4.1 ModelFallbackMiddleware——模型故障自动切换

**解决的问题：** 主模型 API 宕机了怎么办？

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4.1",            # 主模型
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4.1-mini",                   # 第一备选
            "claude-3-5-sonnet-20241022",      # 第二备选
        ),
    ],
)
```

**执行逻辑：**

```
调用主模型 gpt-4.1
  ├── 成功 → 使用结果
  └── 失败 → 调用第一备选 gpt-4.1-mini
                ├── 成功 → 使用结果
                └── 失败 → 调用第二备选 claude-3-5-sonnet
                              ├── 成功 → 使用结果
                              └── 失败 → 抛出异常
```

**使用建议：** 把便宜的或不同提供商的模型作为备选，兼顾成本和可用性

### 4.2 ModelRetryMiddleware——模型调用失败自动重试

**解决的问题：** 模型 API 偶尔超时或返回 429（限流）错误，简单重试即可解决

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,         # 最多重试 3 次（加上首次共 4 次尝试）
            backoff_factor=2.0,    # 指数退避因子
            initial_delay=1.0,     # 初始等待 1 秒
            max_delay=60.0,        # 最长等待 60 秒
            jitter=True,           # 随机抖动，避免多个请求同时重试
        ),
    ],
)
```

**指数退避（Exponential Backoff）的工作方式：**

```
第 1 次调用: 立即执行
  → 失败
第 1 次重试: 等待 1.0 秒（initial_delay × backoff_factor^0）
  → 失败
第 2 次重试: 等待 2.0 秒（1.0 × 2.0^1）
  → 失败
第 3 次重试: 等待 4.0 秒（1.0 × 2.0^2）
  → 失败
所有重试用尽 → 执行 on_failure 策略
```

**`on_failure` 策略：**

```python
# 方式一（默认）：返回包含错误信息的 AIMessage，让 Agent 尝试继续
on_failure="continue"

# 方式二：直接抛出异常，中断 Agent 执行
on_failure="error"

# 方式三：自定义函数，返回自定义错误消息
def format_error(error: Exception) -> str:
    return f"模型调用失败: {error}，请稍后重试。"
on_failure=format_error
```

**只重试特定类型的错误：**

```python
# 只在超时和连接错误时重试
ModelRetryMiddleware(
    max_retries=3,
    retry_on=(TimeoutError, ConnectionError),  # 其他错误直接抛出
)

# 用函数判断是否重试
def should_retry(error: Exception) -> bool:
    if hasattr(error, "status_code"):
        return error.status_code in (429, 503)  # 只重试限流和服务不可用
    return False

ModelRetryMiddleware(retry_on=should_retry)
```

### 4.3 ToolRetryMiddleware——工具调用失败自动重试

**与 ModelRetry 的区别：** ModelRetry 重试的是模型 API 调用，ToolRetry 重试的是工具函数执行（如外部 API 调用）。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool, api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            tools=["api_tool"],                    # 只对 api_tool 启用重试
            retry_on=(ConnectionError, TimeoutError),  # 只重试网络错误
            on_failure="return_message",           # 失败后返回错误消息给模型
        ),
    ],
)
```

**`on_failure` 策略与 ModelRetry 略有不同：**

```python
# 返回 ToolMessage 包含错误信息，让模型自行处理
on_failure="return_message"  # 默认

# 直接抛出异常
on_failure="raise"

# 自定义错误消息
on_failure=lambda e: f"工具执行失败: {e}"
```

### 4.4 三种容错中间件的对比

| 中间件 | 作用对象 | 失败处理方式 | 典型场景 |
|-------|---------|------------|---------|
| ModelFallback | 模型调用 | 切换到备用模型 | 提供商宕机 |
| ModelRetry | 模型调用 | 等待后重试同一模型 | 临时网络波动、限流 |
| ToolRetry | 工具执行 | 等待后重试同一工具 | 外部 API 超时 |

**它们可以组合使用——先重试，重试失败后切换备用：**

```python
agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool],
    middleware=[
        ModelRetryMiddleware(max_retries=2),       # 先重试 2 次
        ModelFallbackMiddleware("gpt-4.1-mini"),   # 都失败后切换备用模型
        ToolRetryMiddleware(max_retries=3),        # 工具失败重试 3 次
    ],
)
```

---

## 第五章：执行控制类中间件

### 5.1 HumanInTheLoopMiddleware——人工审批

**解决的问题：** 某些工具操作风险高（如发送邮件、数据库写入、转账），需要人类确认后才能执行。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4.1",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),  # 必须有 checkpointer！
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,  # 读邮件不需要审批
            }
        ),
    ],
)
```

**执行流程：**

```
用户: "帮我给张三发一封邮件，主题是项目进度"

Agent 思考 → 决定调用 send_email_tool
  ↓
[HumanInTheLoopMiddleware 拦截]
  → 暂停执行
  → 向人类展示: "Agent 想要调用 send_email_tool，参数是..."
  ↓
人类审核:
  ├── approve  → 继续执行工具
  ├── edit     → 修改参数后执行（如改收件人）
  └── reject   → 取消这次工具调用
```

**为什么需要 checkpointer？** 中断执行后，Agent 的当前状态需要被保存，等人类审核完毕后才能从断点恢复。没有 checkpointer 就无法保存和恢复状态。

**与第六节（Streaming）的结合：** 第六节讲过流式输出中的 interrupt 事件就来自这个中间件。`updates` 流中的 `__interrupt__` 事件表示 Agent 被中断等待审批。

### 5.2 ModelCallLimitMiddleware——模型调用次数限制

**解决的问题：** 防止 Agent 进入死循环，无限制地调用模型。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4.1",
    checkpointer=InMemorySaver(),  # thread_limit 需要 checkpointer
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,      # 整个对话线程最多调用模型 10 次
            run_limit=5,          # 单次 invoke/stream 最多调用模型 5 次
            exit_behavior="end",  # 达到限制时优雅结束（另一选项: "error" 抛异常）
        ),
    ],
)
```

**两种限制维度：**

```
thread_limit（线程级）:
  用户消息1 → 模型调用×2  ← 累计 2
  用户消息2 → 模型调用×3  ← 累计 5
  用户消息3 → 模型调用×4  ← 累计 9
  用户消息4 → 模型调用×1  ← 累计 10，达到 thread_limit！

run_limit（单次调用级）:
  用户消息1 → 模型调用×5  ← 达到 run_limit！本次结束
  用户消息2 → 模型调用×3  ← 重新从 0 计数（run_limit 每次重置）
```

### 5.3 ToolCallLimitMiddleware——工具调用次数限制

**与 ModelCallLimit 的区别：** ModelCallLimit 限制模型被调用的次数，ToolCallLimit 限制工具被调用的次数。而且 ToolCallLimit 可以针对**特定工具**设限。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool, scraper_tool],
    middleware=[
        # 全局限制：所有工具总共最多调用 20 次
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),

        # 针对特定工具的限制
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,         # search 工具最多调用 5 次
            run_limit=3,            # 单次最多 3 次
        ),

        # 严格限制：超过立即报错
        ToolCallLimitMiddleware(
            tool_name="scrape_webpage",
            run_limit=2,
            exit_behavior="error",  # 超过直接抛异常
        ),
    ],
)
```

**三种 `exit_behavior`：**

| 行为 | 说明 | 适用场景 |
|------|------|---------|
| `"continue"`（默认） | 阻止超限的工具调用，但 Agent 继续运行 | 一般场景，让模型自己决定下一步 |
| `"error"` | 直接抛出异常，中断执行 | 严格限制，绝对不能超调 |
| `"end"` | 立即结束，返回一条提示消息 | 单工具场景 |

### 5.4 LLMToolSelectorMiddleware——智能工具筛选

**解决的问题：** Agent 有很多工具（10+），但每次请求通常只需要其中几个。把所有工具的描述都发给模型既浪费 token，又可能让模型"分心"。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[tool1, tool2, tool3, tool4, tool5, ...],  # 很多工具
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4.1-mini",      # 用便宜的模型做筛选
            max_tools=3,               # 最多选 3 个工具
            always_include=["search"], # search 工具永远保留
        ),
    ],
)
```

**工作流程：**

```
用户: "查一下明天北京的天气"

第一步（LLMToolSelector 工作）:
  → 把用户消息 + 所有工具描述发给 gpt-4.1-mini
  → gpt-4.1-mini 返回: ["get_weather", "search"]  ← 筛选出相关工具

第二步（Agent 正常工作）:
  → 只把 get_weather 和 search 两个工具传给主模型 gpt-4.1
  → 主模型正常思考和调用工具

效果:
  原来传 50 个工具的描述（~5000 tokens）
  现在只传 2 个工具的描述（~200 tokens）
  → 节省 token + 提高准确率
```

---

## 第六章：监控与安全类中间件

### 6.1 PIIMiddleware——敏感信息检测

**PII = Personally Identifiable Information（个人可识别信息），** 如邮箱、信用卡号、IP 地址、身份证号等。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        # 检测邮箱并脱敏
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # 检测信用卡号并遮盖
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
```

**四种处理策略（strategy）：**

```
原始文本: "我的邮箱是 zhang@example.com，信用卡号是 4111-1111-1111-1234"

"block"（阻止）:
  → 直接拒绝处理，抛出异常
  → 适用于绝对不允许出现敏感信息的场景

"redact"（脱敏替换）:
  → "我的邮箱是 [REDACTED_EMAIL]，信用卡号是 [REDACTED_CREDIT_CARD]"
  → 完全不可逆，信息彻底消失

"mask"（部分遮盖）:
  → "我的邮箱是 ****@*****.**m，信用卡号是 ****-****-****-1234"
  → 保留部分信息用于核实

"hash"（哈希替换）:
  → "我的邮箱是 a1b2c3d4...，信用卡号是 e5f6g7h8..."
  → 确定性的——同一个邮箱总是产生相同的哈希值（可用于数据关联分析）
```

**检测范围控制：**

```python
PIIMiddleware(
    "email",
    strategy="redact",
    apply_to_input=True,          # 检查用户输入（默认开启）
    apply_to_output=False,        # 检查模型输出
    apply_to_tool_results=False,  # 检查工具返回结果
)
```

**内置 PII 类型：** `email`、`credit_card`、`ip`、`mac_address`、`url`

**自定义 PII 类型——** 通过 `detector` 参数扩展检测能力：

```python
import re

# 方式一：正则表达式字符串
PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block")

# 方式二：编译后的正则
PIIMiddleware("phone", detector=re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}"), strategy="mask")

# 方式三：自定义函数（最灵活）
def detect_ssn(content: str) -> list[dict[str, str | int]]:
    """检测社会安全号码，带验证逻辑。"""
    matches = []
    for match in re.finditer(r"\d{3}-\d{2}-\d{4}", content):
        ssn = match.group(0)
        first_three = int(ssn[:3])
        # 验证：前三位不能是 000、666 或 900-999
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({
                "text": ssn,
                "start": match.start(),
                "end": match.end(),
            })
    return matches

PIIMiddleware("ssn", detector=detect_ssn, strategy="hash")
```

**自定义检测函数的返回格式：**

```python
def detector(content: str) -> list[dict[str, str | int]]:
    return [
        {"text": "匹配到的文本", "start": 起始位置, "end": 结束位置},
        ...
    ]
```

---

## 第七章：能力扩展类中间件

### 7.1 TodoListMiddleware——任务规划与跟踪

**解决的问题：** 复杂的多步骤任务中，Agent 需要一个"清单"来规划和跟踪进度。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)
```

这个中间件会自动给 Agent 添加一个 `write_todos` 工具和相应的系统提示。Agent 可以用这个工具来：

- 列出待完成的任务
- 标记已完成的步骤
- 调整任务优先级

### 7.2 ShellToolMiddleware——命令行执行

**解决的问题：** 让 Agent 能执行系统命令（如运行脚本、安装依赖、操作文件等）。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            execution_policy=HostExecutionPolicy(),  # 直接在主机上执行
        ),
    ],
)
```

**三种执行策略（安全级别递增）：**

| 策略 | 安全级别 | 说明 |
|------|---------|------|
| `HostExecutionPolicy` | 低 | 直接在主机上执行，最灵活但最不安全 |
| `DockerExecutionPolicy` | 中 | 在 Docker 容器中执行，硬隔离 |
| `CodexSandboxExecutionPolicy` | 高 | 在沙箱中执行，系统调用受限 |

**Docker 隔离 + 启动命令示例：**

```python
from langchain.agents.middleware import ShellToolMiddleware, DockerExecutionPolicy

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            startup_commands=["pip install requests", "export PYTHONPATH=/workspace"],
            execution_policy=DockerExecutionPolicy(
                image="python:3.11-slim",
                command_timeout=60.0,  # 单条命令超时 60 秒
            ),
        ),
    ],
)
```

### 7.3 FileSearchMiddleware——文件搜索

提供 glob（文件名匹配）和 grep（内容搜索）两个工具：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/workspace",
            use_ripgrep=True,       # 使用 ripgrep 提高搜索速度
            max_file_size_mb=10,    # 跳过大于 10MB 的文件
        ),
    ],
)

# Agent 可以使用:
# glob_search(pattern="**/*.py")           → 找到所有 Python 文件
# grep_search(pattern="async def", include="*.py")  → 搜索包含 async def 的文件
```

### 7.4 FilesystemMiddleware——文件系统读写

来自 Deep Agents 库，提供四个文件操作工具（`ls`、`read_file`、`write_file`、`edit_file`），支持**短期文件系统**和**长期持久化存储**：

```python
from langchain.agents import create_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create_agent(
    model="claude-sonnet-4-6",
    store=store,
    middleware=[
        FilesystemMiddleware(
            backend=CompositeBackend(
                default=StateBackend(),            # 默认：短期存储（在图状态中）
                routes={"/memories/": StoreBackend()}  # /memories/ 路径：持久化存储
            ),
        ),
    ],
)
```

**短期 vs 长期存储：**

```
写入 /notes/task.txt      → StateBackend（短期，当前会话有效）
写入 /memories/user.txt   → StoreBackend（持久化，跨会话存活）
```

### 7.5 SubAgentMiddleware——子Agent委派

**解决的问题：** 主Agent遇到复杂子任务时，委派给专门的子Agent处理，保持主Agent上下文干净。

```python
from langchain.tools import tool
from langchain.agents import create_agent
from deepagents.middleware.subagents import SubAgentMiddleware


@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city} 今天是晴天。"


agent = create_agent(
    model="claude-sonnet-4-6",
    middleware=[
        SubAgentMiddleware(
            default_model="claude-sonnet-4-6",
            default_tools=[],
            subagents=[
                {
                    "name": "weather",
                    "description": "这个子Agent可以查询城市天气。",
                    "system_prompt": "使用 get_weather 工具获取天气。",
                    "tools": [get_weather],
                    "model": "gpt-4.1",     # 子Agent可以用不同的模型
                    "middleware": [],        # 子Agent也可以有自己的中间件
                }
            ],
        )
    ],
)
```

**工作原理：**

```
用户: "帮我查北京天气，然后写一段出行建议"

主Agent (supervisor):
  → 委派给 weather 子Agent: "查北京天气"
  ← 收到简洁结果: "北京今天晴天"
  → 自己写出行建议（不需要看到子Agent内部的工具调用细节）
```

**核心价值——上下文隔离：** 子Agent内部可能调用了多个工具、生成了大量中间消息，但主Agent只看到最终结果。这避免了主Agent的上下文被无关信息"污染"。

每个Agent还自动拥有一个 `general-purpose` 子Agent（与主Agent能力相同），专门用于上下文隔离——把复杂任务委派给它，得到简洁结果。

### 7.6 LLMToolEmulator——用LLM模拟工具（测试用）

**解决的问题：** 开发阶段，真实工具还没实现或调用成本高，用 LLM 生成模拟结果来测试 Agent 的整体行为。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator

# 模拟所有工具
agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator()],  # 所有工具调用都被 LLM 模拟
)

# 只模拟特定工具
agent2 = create_agent(
    model="gpt-4.1",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(tools=["send_email"])],  # 只模拟 send_email
)
```

**使用场景示例：**

```
正常执行:
  Agent → 调用 send_email(to="张三", subject="进度") → 真的发送邮件

使用 LLMToolEmulator:
  Agent → 调用 send_email(to="张三", subject="进度")
        → LLM 生成模拟结果: "Email sent to 张三 with subject '进度'"
        → 邮件并没有真的发出（安全的测试环境）
```

---

## 第八章：中间件的组合策略

### 8.1 组合顺序的考量

中间件的顺序会影响行为。一般建议的组合顺序是：

```python
agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    checkpointer=InMemorySaver(),
    middleware=[
        # 第一层：安全检查（最先执行，最后返回）
        PIIMiddleware("email", strategy="redact"),

        # 第二层：上下文管理
        SummarizationMiddleware(model="gpt-4.1-mini", trigger=("tokens", 4000)),
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100000)]),

        # 第三层：执行控制
        LLMToolSelectorMiddleware(model="gpt-4.1-mini", max_tools=5),
        ModelCallLimitMiddleware(run_limit=10),
        ToolCallLimitMiddleware(tool_name="search", run_limit=5),
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # 第四层：容错
        ModelRetryMiddleware(max_retries=2),
        ModelFallbackMiddleware("gpt-4.1-mini"),
        ToolRetryMiddleware(max_retries=3),

        # 第五层：能力扩展
        TodoListMiddleware(),
    ],
)
```

### 8.2 常见组合模式

**生产环境基础套件：**

```python
middleware=[
    PIIMiddleware("email", strategy="redact"),
    SummarizationMiddleware(model="gpt-4.1-mini", trigger=("tokens", 8000)),
    ModelCallLimitMiddleware(run_limit=20),
    ModelRetryMiddleware(max_retries=2),
    ModelFallbackMiddleware("gpt-4.1-mini"),
]
```

**高安全场景（涉及敏感操作）：**

```python
middleware=[
    PIIMiddleware("email", strategy="block"),
    PIIMiddleware("credit_card", strategy="block"),
    HumanInTheLoopMiddleware(interrupt_on={"transfer_money": True}),
    ModelCallLimitMiddleware(run_limit=5),
    ToolCallLimitMiddleware(run_limit=3),
]
```

**开发测试套件：**

```python
middleware=[
    LLMToolEmulator(tools=["send_email", "database_write"]),  # 模拟危险工具
    TodoListMiddleware(),
    ModelCallLimitMiddleware(run_limit=10),  # 防止无限循环
]
```

---

## 第九章：概念串联——中间件在 Agent 生态中的位置

### 9.1 与前几节知识的关联

```
第一节（Agent）
  → 中间件在 Agent 核心循环的关键节点插入钩子
  → create_agent 的 middleware 参数是入口
  → ReAct 循环的 "模型调用" 和 "工具执行" 对应中间件的两大类钩子

第二节（Model）
  → ModelFallback 切换不同模型
  → ModelRetry 重试模型调用
  → LLMToolSelector 用轻量模型预筛选工具
  → Summarization 用便宜模型做摘要

第三节（Messages）
  → 中间件通过 ToolMessage 向模型传递错误信息和重试反馈
  → PII 中间件修改消息内容（脱敏/替换）
  → ContextEditing 修改旧 ToolMessage 的内容

第四节（Tools）
  → ToolRetry 重试工具调用
  → ToolCallLimit 限制工具调用次数
  → LLMToolSelector 在模型调用前筛选可用工具
  → TodoList/Shell/FileSearch 通过中间件添加新工具

第五节（Short-term Memory）
  → Summarization 压缩历史（memory 溢出时的解决方案）
  → ContextEditing 清理旧工具输出
  → HumanInTheLoop 需要 checkpointer 保存中断状态

第六节（Streaming）
  → HumanInTheLoop 的中断事件通过 updates 流的 __interrupt__ 传递
  → 中间件的行为可以通过 custom 流模式向前端传递进度

第七节（Structured Output）
  → LLMToolSelector 内部使用结构化输出让筛选模型返回工具列表
  → ToolStrategy 的错误重试机制与中间件的重试逻辑类似

本节（Middleware）← 你在这里
  → 在 Agent 循环的每个环节插入控制逻辑
  → 不修改核心代码，通过组合实现复杂行为
  → 解耦、可复用、可组合
```

---

## 第十章：速查手册

### 中间件功能速查

| 中间件 | 一句话说明 | 需要 checkpointer |
|:-----:|:--------:|:----------------:|
| SummarizationMiddleware | 自动压缩长对话 | 否 |
| ContextEditingMiddleware | 清理旧工具输出 | 否 |
| ModelFallbackMiddleware | 主模型挂了切备用 | 否 |
| ModelRetryMiddleware | 模型调用失败重试 | 否 |
| ToolRetryMiddleware | 工具调用失败重试 | 否 |
| HumanInTheLoopMiddleware | 人工审批工具调用 | 是 |
| ModelCallLimitMiddleware | 限制模型调用次数 | thread_limit 需要 |
| ToolCallLimitMiddleware | 限制工具调用次数 | thread_limit 需要 |
| LLMToolSelectorMiddleware | 智能筛选相关工具 | 否 |
| PIIMiddleware | 敏感信息检测处理 | 否 |
| TodoListMiddleware | 任务规划跟踪 | 否 |
| ShellToolMiddleware | 命令行执行 | 否 |
| FilesystemFileSearchMiddleware | 文件搜索 | 否 |
| FilesystemMiddleware | 文件读写+持久化 | 否 |
| SubAgentMiddleware | 子Agent委派 | 否 |
| LLMToolEmulator | 模拟工具执行（测试） | 否 |

### 指数退避参数速查（ModelRetry / ToolRetry 共用）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `max_retries` | 2 | 重试次数（不含首次） |
| `backoff_factor` | 2.0 | 退避倍数（0.0=固定延迟） |
| `initial_delay` | 1.0秒 | 首次重试等待时间 |
| `max_delay` | 60.0秒 | 最长等待时间 |
| `jitter` | True | 随机抖动（±25%） |

### PII 处理策略速查

| strategy | 效果 | 可逆性 |
|----------|------|--------|
| `"block"` | 拒绝处理，抛异常 | — |
| `"redact"` | 替换为 `[REDACTED_TYPE]` | 不可逆 |
| `"mask"` | 部分遮盖 `****1234` | 不可逆 |
| `"hash"` | 确定性哈希值 | 不可逆但可关联 |
