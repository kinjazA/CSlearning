# 01 什么是Agent

## 01.1 聊天bot vs Agent

**普通聊天机器人**就像一个只会"回答问题"的人。你问它"今天天气怎么样？"，它只能从训练数据里找答案，如果训练数据里没有今天的天气，它就回答不了，或者瞎编

**Agent（智能代理）** 则像一个"会使用工具的人"。你问它"今天天气怎么样？"，它会想："我自己不知道天气，但我有一个天气查询工具，让我调用它来查一下。"然后它调用工具，拿到结果，再把结果组织成自然语言告诉你

## 01.2 Agent工作流程

Agent 的核心是一个"思考-行动"循环：

```css
用户提问
   ↓
模型思考："我需要调用工具吗？调用哪个？"
   ↓
┌─ 如果需要 → 调用工具 → 拿到结果 → 再次思考（可能需要调用更多工具）
└─ 如果不需要 → 直接回答
   ↓
最终回答用户
```

这个循环可以重复多次。比如用户问"我这里的天气怎么样？"，Agent 可能会：

1. 第一轮思考：用户说"我这里"，但我不知道用户在哪 → 调用 `get_user_location` 工具
2. 拿到结果："Florida"
3. 第二轮思考：现在知道用户在 Florida 了，查天气 → 调用 `get_weather("Florida")`
4. 拿到结果："It's always sunny in Florida!"
5. 第三轮思考：信息足够了，组织回答
6. 最终回答用户

## 01.3 Langchain的作用

LangChain 是一个 **Python 框架**，它把"大语言模型"和"工具"和"记忆"等组件**粘合**在一起，不用从零写复杂的调度逻辑。只需要：

- 定义工具（Python 函数）
- 选择模型（比如 Claude）
- 写好提示词
- 调用 `create_agent()` 把它们组装起来

LangChain 会自动处理"模型决策→调用工具→处理结果→再次决策"这个复杂的循环

# 02 环境准备

## 02.1 安装

一系列Langchain包，现在推荐用`uv`来做项目环境管理

```bash
uv add langchain
# Requires Python 3.10+

# Installing the OpenAI integration
uv add langchain-openai

# Installing the Anthropic integration
uv add langchain-anthropic

# 后续如果用的是别家的大模型，这里还要进一步安装对应的包
```

## 02.2 获取API Key

API Key 是访问大模型服务的"密码"。需要：

1. 访问 Anthropic 官网（https://console.anthropic.com）
2. 注册账号
3. 在控制台中创建一个 API Key（一串类似 `sk-ant-xxxxx` 的字符串）

## 02.3 设置环境变量

在终端中运行：

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-你的密钥"

# Windows（命令提示符）
set ANTHROPIC_API_KEY=sk-ant-你的密钥

# Windows（PowerShell）
$env:ANTHROPIC_API_KEY="sk-ant-你的密钥"
```

**为什么用环境变量？** 若把 API Key 直接写在代码里不安全（别人看到你的代码就能盗用 Key）。设置为环境变量后，LangChain 会自动从系统中读取，代码里不需要硬编码密钥，更安全

## 02.4 模型选择

文档用的是 Anthropic 的 Claude，但 LangChain 支持多种模型。如果想换用其他模型，只需要：

1. 安装对应的包（比如 `pip install langchain-openai`）
2. 设置对应的环境变量（比如 `OPENAI_API_KEY`）
3. 在代码中改模型名（比如 `"gpt-4o"`）

==这里要根据买的大模型，去看他家的文档说明来操作，但大部分是兼容openai家的格式的SDK==

# 03 基础Agent快速入门

## 03.1 定义工具函数

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"   # 这里仅是一个模拟工具的函数

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

**`city: str`**——参数和类型注解。`city` 是参数名，`: str` 告诉 Python（和模型）这个参数应该是字符串类型。模型会根据这个信息知道"调用这个工具时，我需要传入一个城市名（字符串）"



**`-> str`**——返回值类型注解。告诉 Python 这个函数返回的是字符串



**`"""Get weather for a given city."""`**——文档字符串（docstring）。这是**最关键的部分**。LangChain 会把这段文字提取出来，作为工具描述发送给模型。模型根据这段描述来判断"什么时候该调用这个工具"的。所以如果写的 docstring 是 `"""这是一个函数"""`，模型就不知道这个工具是干什么的，就不会在合适的时机调用它



**`return f"It's always sunny in {city}!"`**——返回值。`f"..."` 是 Python 的 f-string，`{city}` 会被替换成实际传入的城市名。这里是假数据，实际项目中会调用真正的天气 API

## 03.2 创建Agent

```python
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
```

`create_agent()` 函数接收三个核心参数，把它们组装成一个可运行的 Agent：

**`model="claude-sonnet-4-6"`**，指定使用哪个大语言模型。这里直接传字符串，LangChain 会自动识别这是 Anthropic 的模型，并使用设置的 `ANTHROPIC_API_KEY` 来连接



**`tools=[get_weather]`**，一个 Python 列表，里面放定义的工具函数。可放多个：`tools=[get_weather, search_web, send_email]`。LangChain 会自动把每个函数的名字、参数、docstring 提取出来，告诉模型"你有这些工具可用"



**`system_prompt="You are a helpful assistant"`**，系统提示词。这段话会在每次对话开始时发送给模型，定义模型的角色和行为准则。模型会始终遵循这个"人设"



**返回值 `agent`**，`create_agent()` 返回一个 Agent 对象。可以把它理解为一个"已经组装好的机器人"，随时可以接收用户消息并产生回复

## 03.3 运行Agent

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

**`agent.invoke(...)`**——调用 Agent 的 `invoke` 方法，意思是"运行一次"

**传入的参数是一个字典（dict）：**

```python
{
    "messages": [
        {"role": "user", "content": "what is the weather in sf"}
    ]
}
```

`"messages"` 是一个消息列表，格式遵循聊天模型的标准：每条消息包含 `role`（角色）和 `content`（内容）。`role` 有三种值：

- `"user"`：用户说的话
- `"assistant"`：AI 说的话
- `"system"`：系统指令（通常由 `system_prompt` 自动处理）

这里只有一条用户消息。如果要传多轮对话历史，可以这样：

```python
{
    "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你？"},
        {"role": "user", "content": "今天天气怎么样？"}
    ]
}
```

## 03.4 Agent内部执行过程

当调用 `agent.invoke(...)` 后，内部发生了这些事：

```css
步骤 1：LangChain 把系统提示词 + 工具描述 + 用户消息打包，发送给 Claude 模型

发送给模型的内容大致如下（简化表示）：
┌─────────────────────────────────────────────┐
│ System: You are a helpful assistant          │
│                                              │
│ Available tools:                             │
│ - get_weather(city: str): Get weather for    │
│   a given city.                              │
│                                              │
│ User: what is the weather in sf              │
└─────────────────────────────────────────────┘

步骤 2：模型分析用户的问题，判断需要调用工具
  模型的思考过程（简化）：
  "用户问 sf 的天气 → 我有 get_weather 工具 → 调用它，参数是 'sf'"

  模型返回的不是普通文本，而是一个工具调用请求：
  {
    "tool": "get_weather",
    "arguments": {"city": "sf"}
  }

步骤 3：LangChain 接收到工具调用请求，执行对应的 Python 函数
  get_weather("sf") → "It's always sunny in sf!"

步骤 4：LangChain 把工具执行结果发回给模型
  "Tool result: It's always sunny in sf!"

步骤 5：模型根据工具结果，生成最终回答
  "The weather in San Francisco is always sunny!"

步骤 6：LangChain 把最终回答返回给你的代码
```

# 04 生产级Agent

基础 Agent 能跑起来，但离生产环境还差很多。下面 6 个步骤，构建一个更完善的 Agent

## 04.1 详细的系统提示词

```python
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell
from the question that they mean wherever they are, use the get_user_location tool to
find their location."""
```

对比基础版的 `"You are a helpful assistant"`，这个提示词做了三件重要的事：

**1. 明确角色定位：** `"You are an expert weather forecaster, who speaks in puns."`告诉模型你是"天气预报专家"，而且要用"双关语/谐音梗"的风格说话。角色越具体，模型的行为越可预测

**2. 列出可用工具：** 虽然 LangChain 会自动把工具信息传给模型，但在提示词中再次说明，能让模型更清楚地理解每个工具的用途

**3. 给出决策指导：** `"If a user asks you for the weather, make sure you know the location..."` 这段话告诉模型一个决策规则：先确认地点，如果用户暗示是"我这里"，就用 `get_user_location` 工具

 ==提示词是控制 Agent 行为的最重要手段。在生产环境中，可能会花大量时间调试提示词，让 Agent 在各种情况下都能做出正确决策==

## 04.2 创建工具

### 1 普通工具

```python
from langchain.tools import tool

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
```

**`@tool` 装饰器的作用：**

在基础版中，我们直接把普通函数传给 `create_agent`。但在生产版中，我们用 `@tool` 装饰器。它做了这些事：

1. **自动提取元数据：** 从函数的名字、参数类型注解、docstring 中提取信息，生成一个标准化的工具描述，让模型能理解
2. **添加额外功能：** 比如参数验证、错误处理、运行时上下文注入
3. **标准化接口：** 让所有工具都遵循统一的接口规范

简单来说，`@tool` 让普通函数"升级"成了 LangChain 能理解和管理的标准化工具

### 2 带上下文的工具

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

**`@dataclass` 和 `class Context`：**`@dataclass` 是 Python 内置装饰器，用来快速创建数据类。`Context` 类定义了运行时上下文的结构——这里只有一个字段 `user_id`（字符串类型）。可以理解为：`Context` 是一个"信封"，里面装着当前用户的信息。在实际项目中，可能还有 `session_id`、`language`、`timezone` 等字段



**`ToolRuntime[Context]`：**`ToolRuntime` 是 LangChain 提供的泛型类型。 表示"这个工具运行时会收到一个包含 `Context` 类型数据的运行时对象"。当模型决定调用 `get_user_location` 时，LangChain 会自动把你在调用 `agent.invoke()` 时传入的 `context` 注入到 `runtime` 参数中



**关键理解：** `runtime` 参数**不是模型传入的**。模型不知道 `runtime` 的存在。LangChain 会自动识别参数中的 `ToolRuntime` 类型，在调用工具时自动注入。模型只会看到这个工具"不需要参数就能获取用户位置"



**`runtime.context.user_id`：**通过 `runtime.context` 访问你传入的 `Context` 对象，再用 `.user_id` 获取用户 ID

**整个流程：**

```css
调用 agent.invoke(..., context=Context(user_id="1"))
     ↓
模型决定调用 get_user_location
     ↓
LangChain 自动注入 runtime，其中 runtime.context = Context(user_id="1")
     ↓
工具函数内部用 runtime.context.user_id 拿到 "1"
     ↓
返回 "Florida"
```

**为什么需要上下文？** 因为有些信息（如当前登录用户的 ID）不应该由模型来决定，而应该由你的应用程序来提供。这保证了安全性和正确性

## 04.3 配置模型

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
```

**`init_chat_model` 函数：**这是 LangChain 的统一模型初始化接口。它根据你传入的模型名，自动选择正确的模型提供商（Anthropic、OpenAI 等）并创建连接。

在基础版中，我们直接传字符串 `model="claude-sonnet-4-6"` 给 `create_agent`，LangChain 内部也是调用 `init_chat_model`。这里显式调用它，是为了设置更多参数。

**参数详解：**

`"claude-sonnet-4-6"`——模型名称。LangChain 会识别前缀来确定提供商：`claude-*` 是 Anthropic，`gpt-*` 是 OpenAI，`gemini-*` 是 Google

`temperature=0.5`——温度参数，控制回答的随机性：

- `0.0`：每次回答完全一样（确定性最高），适合需要精确答案的场景
- `0.5`：适度随机，平衡准确性和创造性
- `1.0`：非常随机（创造性最高），适合头脑风暴或创意写作

对于天气预报这种需要事实准确但表达有趣的场景，0.5 是个好选择

`timeout=10`——超时时间，单位是秒。如果模型在 10 秒内没有返回结果，就抛出超时错误。这在生产环境中很重要，防止用户无限等待

`max_tokens=1000`——模型回答的最大 token 数。1 个 token 大约等于 0.75 个英文单词或 0.5 个中文字。设置上限可以控制成本和响应时间

==这里要注意，像qwen系列模型，还不支持init_chat_model，可能还是要去像是`Langchain.community`里面去导入通义千问的对应的包==

## 04.4 结构化输出

```python
from dataclasses import dataclass

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None
```

默认情况下，模型返回的是自由文本，比如 `"佛罗里达今天阳光明媚！"`。但在实际应用中，代码需要可靠地处理模型的输出。比如你可能需要：

- 把天气数据显示在 UI 的特定位置
- 把天气状况存入数据库
- 根据天气条件触发不同的后续逻辑

如果模型返回的是自由文本，就需要写复杂的解析逻辑（而且经常出错）。结构化输出让模型直接返回一个固定格式的对象

**字段解释：**

`punny_response: str`——必填字段，类型是字符串。存放带双关语风格的回答文本。没有默认值，意味着模型必须提供这个字段

`weather_conditions: str | None = None`——可选字段。`str | None` 表示可以是字符串或 None（空值）。`= None` 是默认值，意味着如果没有天气信息（比如用户只是说"谢谢"），这个字段可以为空

**模型如何遵循这个格式？**LangChain 会把 `ResponseFormat` 的结构转化为模型可理解的指令，告诉模型："你的回答必须包含 `punny_response`（必填）和 `weather_conditions`（选填）这两个字段。"模型会按要求生成符合格式的 JSON，LangChain 再把 JSON 转换为 Python 的 `ResponseFormat` 对象

## 04.5 添加记忆

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

**什么是 Checkpointer（检查点保存器）？**

每次和 Agent 对话时，Agent 需要记住之前说过什么。`Checkpointer` 就是负责保存和恢复对话历史的组件



**`InMemorySaver`：**把对话历史保存在 Python 进程的内存中。优点是速度快、使用简单；缺点是程序一重启，所有对话记录就丢失了

**Checkpointer 和 thread_id 的关系：**

`Checkpointer` 保存的是按 `thread_id` 分组的对话历史。不同的 `thread_id` 代表不同的对话，互不干扰。同一个 `thread_id` 下的多次调用会共享历史记录。

```css
thread_id = "1"  →  对话记录 A（用户张三和 Agent 的对话）
thread_id = "2"  →  对话记录 B（用户李四和 Agent 的对话）
thread_id = "3"  →  对话记录 C（张三的另一个对话主题）
```

## 04.6 组装并运行

### 1 创建Agent

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)
```

**参数逐个解析：**

`model=model`，传入步骤 3 中配置好的模型对象（不再是字符串，而是一个带有 temperature 等配置的对象）



`system_prompt=SYSTEM_PROMPT`，传入步骤 1 中定义的详细系统提示词



`tools=[get_user_location, get_weather_for_location]`，工具列表。注意这里有两个工具，Agent 可以根据需要选择调用哪个（或两个都调用）



`context_schema=Context`，告诉 Agent "运行时上下文的格式是 `Context` 类"。这样 LangChain 知道如何将你传入的上下文注入到工具中

`response_format=ToolStrategy(ResponseFormat)`，指定输出格式



**什么是 `ToolStrategy`？**`ToolStrategy` 是一种让模型生成结构化输出的策略。它的工作原理是：把 `ResponseFormat` 也包装成一个"工具"，让模型通过"调用工具"的方式来生成结构化数据。这比直接要求模型输出 JSON 更可靠，因为模型调用工具时本身就需要传入结构化参数



`checkpointer=checkpointer`，传入步骤 5 中创建的记忆保存器

### 2 运行Agent

```python
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)
```

**`config` 字典：**

```python
config = {"configurable": {"thread_id": "1"}}
```

`config` 是一个配置字典，目前最重要的配置是 `thread_id`。它是对话线程的唯一标识符，可以是任何字符串（如 `"1"`、`"user-123-chat-456"` 等）。同一个 `thread_id` 下的所有调用共享对话历史。

**`agent.invoke(...)` 的三个参数：**

第一个参数——消息字典：

```python
{"messages": [{"role": "user", "content": "what is the weather outside?"}]}
```

和基础版一样，传入用户的消息。

第二个参数——`config=config`：

传入配置，让 Agent 知道这属于哪个对话线程，以及从 checkpointer 中加载/保存哪个对话的历史。

第三个参数——`context=Context(user_id="1")`：

创建一个 `Context` 实例，`user_id` 设为 `"1"`。这个上下文会被注入到需要它的工具中（即 `get_user_location`）。

### 3 读取回答

```python
print(response['structured_response'])
```

`response` 是一个字典，其中 `'structured_response'` 包含按 `ResponseFormat` 格式化的结果：

```python
ResponseFormat(
    punny_response="Florida is still having a 'sun-derful' day! ...",
    weather_conditions="It's always sunny in Florida!"
)
```

可以像访问对象属性一样使用它：

```python
result = response['structured_response']
print(result.punny_response)        # 获取双关语回答
print(result.weather_conditions)     # 获取天气状况
```

这就是结构化输出的优势，不需要从自由文本中"解析"信息，直接通过属性名就能拿到要的数据

### 4 验证记忆

```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! ...",
#     weather_conditions=None   # ← 注意这里是 None
# )
```

**为什么这次 `weather_conditions` 是 `None`？**

因为用户只是说了"谢谢"，没有问天气。模型判断不需要调用天气工具，所以没有天气信息要填，该字段为 `None`（我们在定义 `ResponseFormat` 时允许了这一点：`str | None = None`）

**记忆在这里起了什么作用？**

注意第二次调用使用了相同的 `config`（同一个 `thread_id`）。这意味着 Agent 能"记住"第一次对话的内容。所以当用户说"谢谢"时，Agent 知道上下文是"之前帮你查了佛罗里达的天气"，回答也是围绕天气话题的。如果你用了不同的 `thread_id`，Agent 就不知道用户在说什么"谢谢"，回答会很困惑

# 05 完整执行流行

```css
用户代码调用：
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config={"configurable": {"thread_id": "1"}},
    context=Context(user_id="1")
)
│
├─ Step 1：LangChain 检查 checkpointer
│  → thread_id="1" 是否有历史记录？
│  → 没有（第一次对话），从空白开始
│
├─ Step 2：LangChain 组装发送给模型的完整消息
│  ┌────────────────────────────────────────────────────────────────┐
│  │ System Prompt:                                                 │
│  │   "You are an expert weather forecaster, who speaks in puns..."│
│  │                                                                │
│  │ Available Tools:                                               │
│  │   1. get_user_location()                                       │
│  │      - Description: Retrieve user information based on user ID │
│  │      - Parameters: 无（runtime 参数对模型不可见）               │
│  │                                                                │
│  │   2. get_weather_for_location(city: str)                       │
│  │      - Description: Get weather for a given city               │
│  │      - Parameters: city (string)                               │
│  │                                                                │
│  │   3. ResponseFormat (作为工具，由 ToolStrategy 注入)            │
│  │      - Parameters: punny_response (string, 必填)               │
│  │                    weather_conditions (string, 选填)            │
│  │                                                                │
│  │ User Message:                                                  │
│  │   "what is the weather outside?"                               │
│  └────────────────────────────────────────────────────────────────┘
│
├─ Step 3：模型第一轮思考
│  模型分析：用户说 "outside"（外面），暗示是当前位置
│  → 但我不知道用户在哪 → 先调用 get_user_location
│
│  模型返回工具调用请求：
│  { "tool": "get_user_location", "arguments": {} }
│
├─ Step 4：LangChain 执行工具
│  → 注入 runtime（其中 context.user_id = "1"）
│  → get_user_location(runtime) 被调用
│  → 函数内部：user_id == "1" → return "Florida"
│  → 工具返回结果："Florida"
│
├─ Step 5：LangChain 把工具结果发回给模型
│  ┌──────────────────────────────────────┐
│  │ Tool Result (get_user_location):     │
│  │   "Florida"                          │
│  └──────────────────────────────────────┘
│
├─ Step 6：模型第二轮思考
│  模型分析：现在知道用户在 Florida → 查 Florida 的天气
│  → 调用 get_weather_for_location
│
│  模型返回工具调用请求：
│  { "tool": "get_weather_for_location", "arguments": {"city": "Florida"} }
│
├─ Step 7：LangChain 执行工具
│  → get_weather_for_location("Florida") 被调用
│  → return "It's always sunny in Florida!"
│
├─ Step 8：LangChain 把工具结果发回给模型
│  ┌────────────────────────────────────────────────┐
│  │ Tool Result (get_weather_for_location):         │
│  │   "It's always sunny in Florida!"               │
│  └────────────────────────────────────────────────┘
│
├─ Step 9：模型第三轮思考
│  模型分析：已经有了所有需要的信息 → 生成最终回答
│  → 调用 ResponseFormat 工具来输出结构化响应
│
│  模型返回：
│  {
│    "tool": "ResponseFormat",
│    "arguments": {
│      "punny_response": "Florida is still having a 'sun-derful' day!...",
│      "weather_conditions": "It's always sunny in Florida!"
│    }
│  }
│
├─ Step 10：LangChain 解析结构化输出
│  → 把 JSON 转换成 ResponseFormat 对象
│
├─ Step 11：LangChain 保存对话历史到 checkpointer
│  → thread_id="1" 的历史中添加这次的完整对话
│  → 包括用户消息、工具调用过程、最终回答
│
└─ Step 12：返回结果给你的代码
   response = {
       "messages": [...完整对话历史...],
       "structured_response": ResponseFormat(
           punny_response="Florida is still having a ...",
           weather_conditions="It's always sunny in Florida!"
       )
   }
```

# 06 核心概念速查

| 概念                | 是什么           | 为什么需要它                                    |
| ------------------- | ---------------- | ----------------------------------------------- |
| `create_agent()`    | Agent 工厂函数   | 把模型、工具、提示词组装成一个可运行的 Agent    |
| `@tool` 装饰器      | 工具标记器       | 把普通 Python 函数升级为标准化的 LangChain 工具 |
| `system_prompt`     | 系统提示词       | 定义 Agent 的角色、行为准则和决策逻辑           |
| `init_chat_model()` | 模型初始化函数   | 创建带配置的模型连接（温度、超时等）            |
| `temperature`       | 温度参数         | 控制回答的随机性/创造性（0=确定，1=随机）       |
| `ToolRuntime`       | 运行时上下文注入 | 让工具访问应用程序级别的信息（如用户 ID）       |
| `Context`           | 上下文数据类     | 定义运行时上下文包含哪些字段                    |
| `ResponseFormat`    | 输出格式定义     | 确保 Agent 返回可预测的结构化数据               |
| `ToolStrategy`      | 输出策略         | 通过工具调用机制来生成结构化输出                |
| `InMemorySaver`     | 内存检查点       | 在内存中保存对话历史（开发测试用）              |
| `thread_id`         | 对话线程标识     | 区分不同的对话，让记忆互不干扰                  |
| `agent.invoke()`    | 运行方法         | 传入消息和配置，触发 Agent 的完整处理流程       |

**Q：工具函数里可以做任何事吗？**

A：是的。工具函数就是普通的 Python 函数，可以在里面调用 API、查数据库、读写文件、发邮件——任何 Python 能做的事。模型只负责决定"什么时候调用"和"传什么参数"，具体执行是你的代码来做

**Q：模型怎么知道该调用哪个工具？**

A：模型根据三个信息来判断：(1) 工具的名字，(2) 工具的描述（docstring），(3) 工具的参数。所以这三样东西的命名和描述要清晰、准确

**Q：如果模型调用工具时传了错误的参数怎么办？**

A：LangChain 会做基本的类型检查。如果参数类型不对，会抛出错误。在生产环境中，应该在工具函数内部做额外的参数验证和错误处理

**Q：`agent.invoke()` 和 `agent.stream()` 有什么区别？**

A：`invoke()` 等所有处理完成后一次性返回结果；`stream()` 则是边处理边返回，用户可以看到实时的"打字"效果。在前端应用中通常用 `stream()` 来提升用户体验