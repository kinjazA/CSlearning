# 01 模型在整个系统中的位置

## 01.1 回顾：模型 = Agent 的大脑

上一节我们用餐厅比喻建立了直觉：Agent 是大堂经理，模型是他的大脑，工具是厨房。但上一节主要讲的是"大堂经理如何工作"（Agent 的运行循环），这一节我们要深入研究"大脑本身"——它能做什么、怎么调用、有哪些高级功能

**模型不仅仅能"聊天"。** 现代大语言模型除了生成文本，还支持：

| 能力 | 说明 | 类比 |
|------|------|------|
| 工具调用（Tool Calling） | 模型可以请求调用外部工具 | 大堂经理吩咐厨师做菜 |
| 结构化输出（Structured Output） | 模型按照你定义的格式返回数据 | 经理用标准表格填写订单 |
| 多模态（Multimodal） | 处理图片、音频、视频等非文本数据 | 经理不仅能听，还能看照片 |
| 推理（Reasoning） | 多步骤推理，展示思考过程 | 经理把自己的思考过程写在纸上给你看 |

## 01.2 两种使用方式

模型可以在两种场景下使用：

```css
方式一：配合 Agent 使用（上一节学过的）
┌──────────────────────────────────┐
│  Agent                           │
│  ┌──────────┐  ┌──────────┐     │
│  │  模型     │  │  工具     │     │ ← 模型作为 Agent 的一部分
│  └──────────┘  └──────────┘     │    Agent 负责循环调度
│  ┌──────────┐                   │
│  │  记忆     │                   │
│  └──────────┘                   │
└──────────────────────────────────┘

方式二：独立使用（本节重点）
┌──────────┐
│  模型     │ ← 直接调用模型，不需要 Agent 框架
└──────────┘   适合简单的文本生成、分类、提取等任务
```

**为什么需要独立使用？** 并不是所有任务都需要 Agent 的完整循环。比如：
- 翻译一句话 → 不需要工具，直接调用模型就行
- 从文本中提取关键信息 → 一次调用就够
- 对内容进行分类 → 不需要多轮推理

LangChain 的模型接口在两种场景下完全一致。你先学会独立使用模型，之后放进 Agent 里时不需要改任何代码

---

#02 初始化模型

## 02.1 `init_chat_model`——通用入口

LangChain 提供了一个统一的函数 `init_chat_model`，只需告诉它模型名字，会自动处理"连接哪个提供商、用什么 SDK"等底层细节。这里最好查一下文档，有些国产llm好像还不支持这个函数

```python
from langchain.chat_models import init_chat_model

# 只需要一行代码，就能初始化一个模型
model = init_chat_model("claude-sonnet-4-6")
```

**这行代码背后发生了什么？**

```css
"claude-sonnet-4-6"
       │
       ▼
LangChain 内部解析：
  1. 名字中包含 "claude" → 识别提供商为 Anthropic
  2. 检查环境变量 ANTHROPIC_API_KEY 是否设置
  3. 自动导入 langchain_anthropic 包
  4. 创建 ChatAnthropic 实例，连接 Anthropic 的 API
       │
       ▼
返回一个可以直接调用的模型对象
```

## 02.2 不同提供商的初始化方式

LangChain 支持所有主流模型提供商。每个提供商有两种初始化方式——**通用方式**和**直接实例化方式**

#### 通用方式：`init_chat_model`（推荐入门）

```python
import os
from langchain.chat_models import init_chat_model

# --- OpenAI ---
os.environ["OPENAI_API_KEY"] = "sk-..."
model = init_chat_model("gpt-5.2")

# --- Anthropic ---
os.environ["ANTHROPIC_API_KEY"] = "sk-..."
model = init_chat_model("claude-sonnet-4-6")

# --- Google Gemini ---
os.environ["GOOGLE_API_KEY"] = "..."
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
#                        ^^^^^^^^^^^^
#                        当模型名无法自动推断提供商时，手动指定前缀
```

#### 直接实例化方式（需要精细控制时）

```python
# --- OpenAI ---
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-5.2")

# --- Anthropic ---
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-4-6")

# --- Google Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
```

**两种方式有什么区别？**

| 对比项 | `init_chat_model` | 直接实例化 |
|--------|-------------------|-----------|
| 代码量 | 少（一行搞定） | 多（需要导入具体类） |
| 提供商切换 | 改个字符串就行 | 需要改导入语句和类名 |
| 参数控制 | 通过 `**kwargs` 传递 | 直接作为构造参数 |
| 适用场景 | 快速原型、需要灵活切换模型 | 生产环境、需要提供商特有功能 |

## 02.3 核心参数详解

无论用哪种方式初始化，都可以通过参数来控制模型行为：

```python
model = init_chat_model(
    "claude-sonnet-4-6",

    # 控制输出的随机性（0~1）
    # 越低越确定，越高越有创意
    temperature=0.7,

    # 模型每次回答的最大 token 数
    # 防止模型写太长、节省成本
    max_tokens=1000,

    # 等待响应的最大秒数
    # 超时后抛出错误，避免程序卡死
    timeout=30,

    # 请求失败时的最大重试次数（默认 6 次）
    # 网络不稳定时可以调高到 10~15
    max_retries=6,
)
```

**`max_retries` 的重试机制值得深入了解：**

```css
请求失败
  │
  ├── 429（请求频率超限）→ 重试 ✓
  ├── 5xx（服务器内部错误）→ 重试 ✓
  ├── 网络超时 → 重试 ✓
  │
  ├── 401（认证失败）→ 不重试 ✗（API Key 错了，重试也没用）
  └── 404（资源不存在）→ 不重试 ✗（地址错了，重试也没用）
```

重试采用**指数退避 + 抖动**策略：第一次等 1 秒，第二次等 2 秒，第三次等 4 秒……同时加入随机偏移，避免大量客户端在同一时刻重试导致"重试风暴"

---

# 03调用模型

模型初始化后，有三种调用方式，适用于不同场景。

## 03.1 `invoke()`——一次性获取完整回答

最直接的方式。发送消息，等模型生成完毕，一次性拿到完整回答

#### 最简单的调用

```python
response = model.invoke("鹦鹉为什么会说话？")
print(response)
# AIMessage(content="鹦鹉会说话是因为...")
```

#### 传入对话历史

模型本身是无状态的——每次调用都是独立的。如果你想让模型"记住"之前的对话，需要**手动把历史消息传给它**

```python
# 方式一：字典格式（简洁、直观）
conversation = [
    # role 表示消息的角色
    {"role": "system", "content": "你是一个翻译助手，将英文翻译为法文。"},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")
```

```python
# 方式二：Message 对象格式（更类型安全）
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("你是一个翻译助手，将英文翻译为法文。"),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
```

**消息角色（role）的完整说明：**

```
┌──────────────┬──────────────────┬──────────────────────────────────┐
│ 角色          │ 对应类            │ 说明                              │
├──────────────┼──────────────────┼──────────────────────────────────┤
│ system       │ SystemMessage    │ 系统指令，定义模型的角色和行为规则    │
│ user         │ HumanMessage     │ 用户（人类）发送的消息               │
│ assistant    │ AIMessage        │ 模型之前的回复（作为历史上下文）      │
│ tool         │ ToolMessage      │ 工具执行的结果（上一节讲过）          │
└──────────────┴──────────────────┴──────────────────────────────────┘
```

**关键理解：模型根据对话历史中的 role 来理解"谁说了什么"。** 你构造的消息列表就像一个剧本，模型读完后接着往下演

## 03.2 `stream()`——边生成边输出

`invoke()` 的问题是：如果模型回答很长，用户需要干等好几秒才能看到内容。`stream()` 解决了这个问题——模型每生成一小段就立即返回，用户可以看到文字"打字机式"地逐步出现

```python
# 基础文本流式输出
for chunk in model.stream("鹦鹉为什么有彩色的羽毛？"):
    print(chunk.text, end="|", flush=True)

# 输出效果（| 表示每个 chunk 的边界）：
# 鹦鹉|有彩色|羽毛|主要是|因为|性选择|...
```

**`chunk` 是什么？**

`invoke()` 返回一个完整的 `AIMessage`，而 `stream()` 返回一系列 `AIMessageChunk`——每个是完整消息的一小段。你可以通过累加把它们拼成完整消息：

```python
full = None  # 初始为空
for chunk in model.stream("天空是什么颜色的？"):
    # 如果是第一个 chunk，直接赋值
    # 否则把新 chunk 加到已有内容上
    full = chunk if full is None else full + chunk
    print(full.text)

# 打印效果——内容逐步增长：
# 天空
# 天空通常
# 天空通常是
# 天空通常是蓝色
# 天空通常是蓝色的
# ...
```

**拼完后的 `full` 可以当作普通的 `AIMessage` 使用**——比如放入对话历史

#### 流式输出中获取工具调用和推理过程

当模型使用了工具调用或推理功能时，流式输出的 chunk 中会包含不同类型的内容块：

```python
for chunk in model.stream("今天天气怎么样？"):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            # := 海象运算符，在表达式里一边赋值，一边使用这个值
            # 推理过程（模型的"思考步骤"）
            print(f"推理中: {reasoning}")
        elif block["type"] == "tool_call_chunk":
            # 工具调用的部分信息（逐步构建中）
            print(f"工具调用片段: {block}")
        elif block["type"] == "text":
            # 普通文本输出
            print(block["text"])
```

## 03.3 `batch()`——批量并行处理

当有多个独立的问题需要处理时，逐个调用 `invoke()` 太慢——每次都要等上一个完成才能开始下一个。`batch()` 可以**并行**发送多个请求：

```python
# 三个问题同时发送，并行处理
responses = model.batch([
    "鹦鹉为什么有彩色的羽毛？",
    "飞机是怎么飞起来的？",
    "什么是量子计算？"
])

# responses 是一个列表，包含三个 AIMessage
for response in responses:
    print(response.text)
```

**控制并行数量：**

```python
# 限制最多同时 5 个并行请求
# 防止触发提供商的速率限制
model.batch(
    questions_list,
    config={
        "max_concurrency": 5,  # 并行上限
    }
)
```

**`batch()` vs `batch_as_completed()`：**

```
batch()：               等所有请求都完成 → 一次性返回全部结果（有序）
batch_as_completed()：  谁先完成就先返回谁的结果（可能无序）
```

```python
# batch_as_completed 的结果可能乱序
# 每个结果包含 input_index 用于匹配原始请求
for response in model.batch_as_completed([
    "问题1", "问题2", "问题3"
]):
    print(response)  # 先完成的先输出
```

**三种调用方式对比：**

| 方式 | 适用场景 | 特点 |
|------|---------|------|
| `invoke()` | 单次请求、需要完整结果 | 最简单；会阻塞直到完成 |
| `stream()` | 面向用户的实时交互 | 用户体验好；需要处理 chunk |
| `batch()` | 离线处理、批量任务 | 效率高；节省总耗时 |

---

# 04 工具调用（Tool Calling）

## 04.1 回顾并深入

上一节已经学了工具的基础概念。这一节从模型视角深入：当模型**独立使用**（不在 Agent 循环中）时，工具调用是怎么运作的？

**核心区别：**

- **在 Agent 中：** Agent 框架会自动帮你执行工具、把结果送回模型——你不需要手动管理这个循环
- **独立使用时：** 需要**自己**编写执行工具和传回结果的逻辑

## 04.2 `bind_tools()`——告诉模型"这些工具可用"

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气信息。"""
    return f"{location}的天气：晴天，25°C"

# bind_tools() 把工具"绑定"到模型上
# 返回一个"带工具"的模型
model_with_tools = model.bind_tools([get_weather])

# 调用时，模型可能不直接回答，而是请求调用工具
response = model_with_tools.invoke("波士顿今天天气怎么样？")

# 检查模型是否请求了工具调用
for tool_call in response.tool_calls:
    print(f"工具名: {tool_call['name']}")   # "get_weather"
    print(f"参数:   {tool_call['args']}")    # {"location": "Boston"}
```

**`bind_tools()` 做了什么？**它把工具的**描述信息**（名字、参数、说明）附加到模型的每次请求中。模型看到这些描述后，就知道自己有哪些工具可以用、什么时候该用

```css
model.invoke("天气怎么样？")
→ 模型不知道有工具 → 只能凭自己的知识回答

model_with_tools.invoke("天气怎么样？")
→ 模型看到 get_weather 工具的描述 → 决定调用它
→ 返回的不是文本回答，而是工具调用请求
```

## 04.3 手动工具执行循环

当不使用 Agent 框架时，你需要自己管理这个循环：

```python
# 第 1 步：绑定工具
model_with_tools = model.bind_tools([get_weather])

# 第 2 步：初始消息
messages = [{"role": "user", "content": "波士顿天气怎么样？"}]

# 第 3 步：模型返回工具调用请求
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)
# 此时 ai_msg.tool_calls = [{"name": "get_weather", "args": {"location": "Boston"}, "id": "call_xxx"}]

# 第 4 步：你来执行工具，把结果加入消息列表
for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)  # ToolMessage 自动包含 tool_call_id

# 第 5 步：把工具结果传回模型，让它生成最终回答
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "波士顿目前的天气是晴天，气温 25°C。"
```

**完整流程图：**

```
用户: "波士顿天气怎么样？"
    │
    ▼
模型收到消息 + 工具描述
    │
    ▼
模型判断："这个问题需要实时天气数据，
          我应该用 get_weather 工具。"
    │
    ▼
模型返回: tool_calls = [{name: "get_weather", args: {location: "Boston"}}]
    │                    （注意：模型没有直接回答，而是发出工具调用请求）
    ▼
你的代码执行: get_weather("Boston")
    │
    ▼
工具返回: "Boston的天气：晴天，25°C"
    │
    ▼
你把结果包装成 ToolMessage，连同之前的消息一起发给模型
    │
    ▼
模型收到完整上下文（用户问题 + 工具结果），生成最终回答:
    "波士顿目前的天气是晴天，气温 25°C。适合户外活动！"
```

**与 Agent 的对比——理解 Agent 的价值：**

上面这个手动循环就是 Agent 内部在做的事情。当你用 `create_agent()` 时，Agent 帮你自动完成了第 3~5 步的循环，而且能处理多轮工具调用。独立使用模型时，这些逻辑需要自己写

## 04.4 并行工具调用

当用户的问题涉及多个独立的查询时，模型可能会同时请求多个工具调用：

```python
response = model_with_tools.invoke(
    "波士顿和东京的天气分别怎么样？"
)

# 模型可能一次返回两个工具调用
print(response.tool_calls)
# [
#   {"name": "get_weather", "args": {"location": "Boston"}, "id": "call_1"},
#   {"name": "get_weather", "args": {"location": "Tokyo"},  "id": "call_2"},
# ]
```

**模型怎么判断是否要并行调用？** 关键在于请求之间是否**互相独立**。波士顿的天气和东京的天气互不影响，所以可以并行。但如果用户说"先查波士顿的天气，然后根据天气推荐穿什么"，模型就会先查天气，等拿到结果后再推荐。

**禁用并行调用：**

```python
# 部分提供商允许关闭并行调用
model.bind_tools([get_weather], parallel_tool_calls=False)
# 模型会一次只调用一个工具
```

## 04.5 强制工具调用

默认情况下，模型自己决定要不要用工具。但有时候你希望模型**必须**调用某个工具：

```python
# 强制模型使用任意一个已绑定的工具（不能跳过工具直接回答）
model.bind_tools([tool_1], tool_choice="any")

# 强制模型使用特定的工具
model.bind_tools([tool_1, tool_2], tool_choice="tool_1")
```

**什么时候需要这个？** 比如你在做一个数据提取管道——用户输入一段文本，你需要模型**必须**调用提取工具来输出结构化数据，而不是自由发挥写一段话。

## 04.6 流式工具调用

工具调用信息也可以流式接收。这在构建实时界面时很有用——你可以在工具调用参数还没完全生成时就开始显示进度：

```python
for chunk in model_with_tools.stream("波士顿和东京天气怎么样？"):
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"工具名: {name}")
        if args := tool_chunk.get("args"):
            print(f"参数片段: {args}")

# 输出——参数是逐步到达的：
# 工具名: get_weather
# 参数片段: {"lo
# 参数片段: catio
# 参数片段: n": "B
# 参数片段: oston"}
# 工具名: get_weather
# 参数片段: {"lo
# 参数片段: cation": "Tokyo"}
```

**注意：** 每个 chunk 中的 `args` 只是 JSON 字符串的一小段，不是完整的 JSON。你需要累加所有 chunk 来获得完整的工具调用信息：

```python
gathered = None
for chunk in model_with_tools.stream("波士顿天气怎么样？"):
    gathered = chunk if gathered is None else gathered + chunk
    print(gathered.tool_calls)  # 逐步构建出完整的 tool_calls 列表
```

---

# 05 结构化输出（Structured Output）

## 05.1 上一节 vs 这一节

上一节我们学了 `ToolStrategy` 和 `ProviderStrategy`——那是**在 Agent 中**使用结构化输出。这一节讲的是**独立使用模型**时的结构化输出方法：`with_structured_output()`。两者的目标相同——让模型输出固定格式的数据——但使用方式不同

## 05.2 `with_structured_output()`——三种 schema 定义方式

#### 方式一：Pydantic 模型（最推荐）

[Pydantic](https://docs.pydantic.dev/) 是 Python 最流行的数据验证库。用它定义输出格式，不仅清晰，还能自动校验数据

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """电影信息。"""
    # Field 的 description 参数会被传给模型，帮助它理解每个字段该填什么
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演姓名")
    rating: float = Field(description="评分，满分 10 分")

# with_structured_output() 返回一个"结构化模型"
# 这个模型的输出会自动解析为 Movie 对象
model_with_structure = model.with_structured_output(Movie)

response = model_with_structure.invoke("告诉我电影《盗梦空间》的信息")
print(response)
# Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)

# 你可以像访问普通对象一样访问字段
print(response.title)     # "Inception"
print(response.year)      # 2010
print(response.rating)    # 8.8
```

**为什么推荐 Pydantic？**
- 自动类型验证：如果模型返回 `year: "两千零十年"`，Pydantic 会报错，你能及时发现问题
- 字段描述会传给模型，提高输出准确性
- 支持嵌套结构（见下文）

#### 方式二：TypedDict（轻量选择）

```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    """电影信息。"""
    title: Annotated[str, ..., "电影标题"]
    year: Annotated[int, ..., "上映年份"]
    director: Annotated[str, ..., "导演姓名"]
    rating: Annotated[float, ..., "评分，满分 10 分"]

model_with_structure = model.with_structured_output(MovieDict)
response = model_with_structure.invoke("告诉我《盗梦空间》的信息")
print(response)
# {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}
```

**与 Pydantic 的区别：** TypedDict 返回的是普通字典（用 `response["title"]` 访问），不做运行时类型验证。适合你不需要严格校验的场景。

#### 方式三：JSON Schema（最灵活）

```python
json_schema = {
    "title": "Movie",
    "description": "电影信息",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "电影标题"},
        "year": {"type": "integer", "description": "上映年份"},
        "director": {"type": "string", "description": "导演姓名"},
        "rating": {"type": "number", "description": "评分，满分 10 分"}
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",  # 指定方法
)
response = model_with_structure.invoke("告诉我《盗梦空间》的信息")
# 返回普通字典
```

**什么时候用 JSON Schema？** 当你的 schema 是从外部系统动态生成的（比如从数据库表结构自动转换），用 JSON Schema 比定义 Python 类更方便。

## 05.3 嵌套结构

实际应用中，数据结构往往是嵌套的。Pydantic 和 TypedDict 都支持嵌套定义：

```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    """演员信息。"""
    name: str              # 演员姓名
    role: str              # 扮演的角色名

class MovieDetails(BaseModel):
    """电影详细信息。"""
    title: str
    year: int
    cast: list[Actor]                              # 演员列表（嵌套！）
    genres: list[str]                              # 类型列表
    budget: float | None = Field(                  # 可选字段
        None, description="预算（百万美元）"
    )

model_with_structure = model.with_structured_output(MovieDetails)
response = model_with_structure.invoke("告诉我《盗梦空间》的详细信息")

# 可以直接访问嵌套对象
for actor in response.cast:
    print(f"{actor.name} 饰演 {actor.role}")
# Leonardo DiCaprio 饰演 Dom Cobb
# Joseph Gordon-Levitt 饰演 Arthur
# ...
```

## 05.4 `method` 参数——结构化输出的实现方式

不同的模型提供商用不同的技术来实现结构化输出。`with_structured_output()` 的 `method` 参数让你选择使用哪种：

```
方法                 原理                                       可靠性
─────────────────────────────────────────────────────────────────────
json_schema         提供商原生的结构化输出功能                      最高
                    （模型被约束只能输出符合 schema 的 JSON）

function_calling    把 schema 伪装成工具调用                       高
                    （就是上一节讲的 ToolStrategy 原理）

json_mode           模型被约束输出合法 JSON                        中
                    （但 schema 需要在 prompt 中描述，
                     模型可能不严格遵守）
```

**选择建议：** 大多数情况下不需要指定 `method`，LangChain 会根据模型提供商自动选择最佳方式

### 5.5 同时获取原始消息和解析结果

有时候你不仅需要结构化数据，还需要模型的原始响应元数据（比如 token 用量）。设置 `include_raw=True`：

```python
model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("告诉我《盗梦空间》的信息")

# response 现在是一个字典，包含三个键
print(response["raw"])           # 原始的 AIMessage 对象（包含 token 用量等元数据）
print(response["parsed"])        # 解析后的 Movie 对象
print(response["parsing_error"]) # 如果解析失败，这里会有错误信息
```

---

# 06 高级主题

## 06.1 模型档案（Model Profiles）

不同的模型有不同的能力——有的支持工具调用，有的不支持；有的上下文窗口很大，有的很小。LangChain 的模型档案功能让你可以**在运行时查询模型的能力**：

```python
print(model.profile)
# {
#   "max_input_tokens": 400000,    # 最大输入 token 数
#   "image_inputs": True,          # 是否支持图片输入
#   "reasoning_output": True,      # 是否支持推理输出
#   "tool_calling": True,          # 是否支持工具调用
#   ...
# }
```

**这有什么用？** 让你的代码可以根据模型能力动态调整行为：

```python
if model.profile.get("tool_calling"):
    # 模型支持工具调用，正常使用
    response = model_with_tools.invoke(query)
else:
    # 模型不支持工具调用，降级为纯文本处理
    response = model.invoke(f"请回答：{query}")
```

## 06.2 多模态（Multimodal）

部分模型可以处理非文本数据。你可以在消息中嵌入图片等内容：

```python
# 发送包含图片的消息（以字典格式为例）
response = model.invoke([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图片里有什么？"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/photo.jpg"}
            }
        ]
    }
])
```

**反过来，模型也可以返回多模态内容：**

```python
response = model.invoke("画一只猫")
print(response.content_blocks)
# [
#     {"type": "text", "text": "这是我画的一只猫"},
#     {"type": "image", "base64": "...", "mime_type": "image/jpeg"},
# ]
```

## 06.3 推理输出（Reasoning）

部分高端模型支持"展示推理过程"——你可以看到模型是怎么一步步思考的，而不仅仅是看到最终答案

```python
response = model.invoke("鹦鹉为什么有彩色的羽毛？")

# 提取推理步骤
reasoning_steps = [
    b for b in response.content_blocks
    if b["type"] == "reasoning"
]

# 打印模型的思考过程
print(" ".join(step["reasoning"] for step in reasoning_steps))
```

**推理 vs 普通输出：**

```
普通输出：
  "鹦鹉有彩色羽毛是因为性选择和伪装。"

推理输出：
  思考1: "用户问的是进化生物学问题..."
  思考2: "彩色羽毛涉及多个进化机制..."
  思考3: "主要因素包括性选择、物种识别、伪装..."
  最终答案: "鹦鹉有彩色羽毛主要是因为性选择和物种识别。"
```

## 06.4 服务端工具调用

有些提供商支持**服务端工具调用**——模型在服务器端直接搜索网页、执行代码，不需要你来编写执行逻辑

```python
# 启用提供商的内置网页搜索工具
tool = {"type": "web_search"}
model_with_tools = model.bind_tools([tool])

response = model_with_tools.invoke("今天有什么好消息？")
print(response.content_blocks)
# [
#     {"type": "server_tool_call", "name": "web_search",
#      "args": {"query": "positive news today"}, "id": "ws_abc123"},
#     {"type": "server_tool_result", "tool_call_id": "ws_abc123",
#      "status": "success"},
#     {"type": "text", "text": "这里是今天的一些好消息...",
#      "annotations": [{"type": "citation", "url": "..."}]}
# ]
```

**与普通工具调用的区别：**

```
普通工具调用（客户端执行）：
  模型请求调用 → 你执行工具 → 你把结果发回模型 → 模型生成回答
  （需要多次 API 往返）

服务端工具调用：
  模型请求调用 → 服务器自动执行 → 模型生成回答
  （一次 API 调用全搞定，没有 ToolMessage 需要你处理）
```

## 06.5 提示词缓存（Prompt Caching）

如果你的系统提示词很长（比如包含一整本书），每次请求都传输同样的内容非常浪费。提示词缓存可以解决这个问题。

**隐式缓存（自动生效）：**
- OpenAI、Gemini 等提供商自动缓存重复的提示词前缀
- 你不需要做任何配置，如果命中缓存就自动省钱

**显式缓存（手动控制）：**
- 你主动标记哪些内容应该被缓存
- 上一节讲的 `cache_control: {"type": "ephemeral"}` 就属于这类

**注意：** 缓存通常需要输入 token 超过一定阈值才会生效。

## 06.6 速率限制（Rate Limiting）

模型提供商通常限制你每分钟能发多少个请求。LangChain 提供了内置的速率限制器，帮你自动控制请求频率：

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

# 每 10 秒最多 1 个请求
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,       # 每秒请求数（0.1 = 每 10 秒 1 个）
    check_every_n_seconds=0.1,     # 每 100ms 检查一次是否可以发送
    max_bucket_size=10,            # 最大突发量
)

model = init_chat_model(
    model="gpt-5",
    rate_limiter=rate_limiter      # 传入速率限制器
)
```

**`max_bucket_size` 是什么？令牌桶算法简述：**

```
想象一个桶，每隔固定时间往里放一个令牌。
每次请求取走一个令牌。
如果桶空了，请求就得等。

max_bucket_size = 10 意味着：
  桶最多存 10 个令牌。
  如果你长时间没发请求，令牌会积累到 10 个。
  这允许你短时间内突发 10 个请求（burst），
  然后回到正常速率。
```

## 06.7 可配置模型（Configurable Models）

你可以创建一个"可配置"的模型——在运行时动态切换底层使用的模型，而不需要重新初始化：

```python
from langchain.chat_models import init_chat_model

# 不指定具体模型，创建一个可配置的模型
configurable_model = init_chat_model(temperature=0)

# 运行时指定用 GPT
configurable_model.invoke(
    "你叫什么名字？",
    config={"configurable": {"model": "gpt-5-nano"}},
)

# 运行时切换到 Claude
configurable_model.invoke(
    "你叫什么名字？",
    config={"configurable": {"model": "claude-sonnet-4-6"}},
)
```

**这比动态模型（上一节讲的 middleware 方式）更轻量。** 适合以下场景：
- A/B 测试不同模型的效果
- 根据用户付费等级使用不同模型
- 在开发和生产环境使用不同模型

**可配置模型也支持工具绑定和结构化输出：**

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """获取天气"""
    location: str = Field(description="城市名")

# 在可配置模型上绑定工具——不管底层用哪个模型都生效
model_with_tools = configurable_model.bind_tools([GetWeather])

# 用 GPT 调用
model_with_tools.invoke(
    "洛杉矶和纽约哪个人口多？",
    config={"configurable": {"model": "gpt-4.1-mini"}}
)

# 用 Claude 调用——同样的工具绑定自动适配
model_with_tools.invoke(
    "洛杉矶和纽约哪个人口多？",
    config={"configurable": {"model": "claude-sonnet-4-6"}}
)
```

## 06.8 调用配置（Invocation Config）

每次调用模型时，你可以通过 `config` 参数传入额外的运行时配置：

```python
response = model.invoke(
    "讲个笑话",
    config={
        "run_name": "joke_generation",       # 本次运行的名称（调试用）
        "tags": ["humor", "demo"],           # 标签（分类、过滤）
        "metadata": {"user_id": "123"},      # 自定义元数据
        "callbacks": [my_callback_handler],  # 回调处理器（监控、日志）
    }
)
```

这些配置在以下场景特别有用：
- **调试追踪：** 在 LangSmith 中查看每次调用的详细信息
- **成本监控：** 通过 metadata 追踪每个用户的 token 消耗
- **生产监控：** 通过 callbacks 记录延迟、错误率等指标

## 06.9 Token 使用量追踪

你可以跟踪模型消耗了多少 token（直接影响 API 费用）：

```python
from langchain_core.callbacks import get_usage_metadata_callback

model_1 = init_chat_model(model="gpt-4.1-mini")
model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

# 使用上下文管理器自动追踪 token 用量
with get_usage_metadata_callback() as cb:
    model_1.invoke("你好")
    model_2.invoke("你好")
    print(cb.usage_metadata)

# 输出：
# {
#   "gpt-4.1-mini-2025-04-14": {
#       "input_tokens": 8, "output_tokens": 10, "total_tokens": 18
#   },
#   "claude-haiku-4-5-20251001": {
#       "input_tokens": 8, "output_tokens": 21, "total_tokens": 29
#   }
# }
```

**为什么要追踪 token？**
- **成本控制：** API 按 token 计费，追踪用量才能控制预算
- **性能优化：** 了解哪些请求消耗最多 token，进行针对性优化
- **异常检测：** 突然的 token 暴增可能意味着提示词注入攻击或 bug

## 06.10 本地模型

LangChain 支持在本地运行模型，不需要联网。推荐使用 [Ollama](https://ollama.ai/)：

```python
from langchain.chat_models import init_chat_model

# Ollama 在本地运行模型
model = init_chat_model(
    "llama3",
    model_provider="ollama",
)

response = model.invoke("你好！")
```

**什么时候用本地模型？**
- 数据隐私要求严格，不能发送到云端
- 需要使用自定义微调的模型
- 想避免 API 调用费用

---

# 07 概念串联——模型在 Agent 中的角色

让我把这一节学到的内容和上一节的 Agent 知识连接起来

## 模型能力 → Agent 行为

```
模型能力                Agent 中的表现
──────────────────────────────────────────────────
invoke()              Agent 每一轮循环都在内部调用 invoke()
stream()              agent.stream() 本质是模型 stream() 的上层包装
tool_calling          Agent 自动管理工具调用循环（上一节第三章）
structured_output     Agent 的 response_format 底层调用 with_structured_output()
multimodal            Agent 可以处理用户上传的图片
reasoning             Agent 可以展示思考过程
rate_limiting         Agent 自动遵守速率限制
```

## 完整的知识地图

```
┌─────────────────────────────────────────────────────────────┐
│                     LangChain Agent                         │
│                                                             │
│  ┌──────────── 本节内容 ────────────┐                       │
│  │          模型（Model）            │                       │
│  │  ┌─────────────────────────┐    │                       │
│  │  │ 初始化                   │    │  ┌─────────────────┐  │
│  │  │ · init_chat_model       │    │  │   工具（Tools）   │  │
│  │  │ · 直接实例化             │    │  │  （上一节第三章） │  │
│  │  └─────────────────────────┘    │  └─────────────────┘  │
│  │  ┌─────────────────────────┐    │                       │
│  │  │ 调用方式                 │    │  ┌─────────────────┐  │
│  │  │ · invoke() 一次性       │    │  │ 中间件           │  │
│  │  │ · stream() 流式         │    │  │ （上一节第八章） │  │
│  │  │ · batch()  批量         │    │  └─────────────────┘  │
│  │  └─────────────────────────┘    │                       │
│  │  ┌─────────────────────────┐    │  ┌─────────────────┐  │
│  │  │ 高级能力                 │    │  │   记忆           │  │
│  │  │ · 工具调用               │    │  │ （上一节第六章） │  │
│  │  │ · 结构化输出             │    │  └─────────────────┘  │
│  │  │ · 多模态                 │    │                       │
│  │  │ · 推理输出               │    │  ┌─────────────────┐  │
│  │  │ · 速率限制               │    │  │  系统提示词       │  │
│  │  │ · 可配置模型             │    │  │ （上一节第四章） │  │
│  │  └─────────────────────────┘    │  └─────────────────┘  │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

# 08 速查手册

## 核心 API 一览

| 你想做什么 | API | 示例 |
|-----------|-----|------|
| 初始化模型（通用） | `init_chat_model()` | `model = init_chat_model("gpt-5")` |
| 初始化模型（指定提供商） | 直接实例化 | `model = ChatOpenAI(model="gpt-5")` |
| 单次调用 | `model.invoke()` | `response = model.invoke("你好")` |
| 流式调用 | `model.stream()` | `for chunk in model.stream("你好"):` |
| 批量调用 | `model.batch()` | `responses = model.batch(["问题1", "问题2"])` |
| 绑定工具 | `model.bind_tools()` | `model_with_tools = model.bind_tools([my_tool])` |
| 结构化输出 | `model.with_structured_output()` | `model.with_structured_output(MySchema)` |
| 查看模型能力 | `model.profile` | `print(model.profile)` |
| 速率限制 | `InMemoryRateLimiter` | `model = init_chat_model(..., rate_limiter=limiter)` |
| 可配置模型 | `config.configurable` | `model.invoke("...", config={"configurable": {"model": "..."}})` |
| Token 追踪 | `get_usage_metadata_callback()` | `with get_usage_metadata_callback() as cb:` |

## 参数速查

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `model` | string | 模型名称 | `"gpt-5"`, `"claude-sonnet-4-6"` |
| `temperature` | float | 随机性（0~1） | 0=精确, 0.5=平衡, 1=创意 |
| `max_tokens` | int | 最大输出 token | 根据任务需要设置 |
| `timeout` | int | 超时秒数 | 30~60 |
| `max_retries` | int | 最大重试次数 | 6（默认）, 不稳定网络用 10~15 |

## 结构化输出方式对比

| 方式 | 返回类型 | 自动校验 | 适用场景 |
|------|---------|---------|---------|
| Pydantic BaseModel | Python 对象 | 有 | 生产环境，需要严格校验 |
| TypedDict | Python 字典 | 无 | 快速开发，轻量需求 |
| JSON Schema | Python 字典 | 无 | Schema 由外部系统生成 |
