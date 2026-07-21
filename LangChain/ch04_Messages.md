# LangChain 消息（Messages）深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第三节。第一节学了 Agent（整体架构），第二节学了 Model（模型/大脑），这一节学习 Messages——模型的"语言"。如果说模型是大脑，那 Messages 就是大脑读和写的"纸条"

---

## 第一章：消息是什么？为什么它是一切的基础？

### 1.1 一个直觉理解

想象你和一个朋友通过纸条交流（不能说话）。每张纸条上需要写三样东西：

1. **谁写的**——是你写的？还是朋友写的？还是"规则说明"？
2. **内容**——纸条上的文字（可能还有贴上去的照片）
3. **附加信息**——比如写的时间、纸条编号

在 LangChain 中，**消息（Message）就是这样的纸条**。你和模型之间的所有交流都是通过消息完成的——你的问题是一条消息，模型的回答也是一条消息，工具的结果也是一条消息

```
消息 = 角色（Role）+ 内容（Content）+ 元数据（Metadata）
```

### 1.2 消息在系统中的位置

```css
┌────────────────────────────────────────────────────┐
│                    LangChain                        │
│                                                     │
│  用户                                               │
│    │                                                │
│    │  HumanMessage("今天天气怎么样？")                │
│    ▼                                                │
│  ┌──────────┐                                       │
│  │  模型     │ ← 接收消息列表，返回 AIMessage         │
│  └──────────┘                                       │
│    │                                                │
│    │  AIMessage(tool_calls=[{name: "get_weather"}]) │
│    ▼                                                │
│  ┌──────────┐                                       │
│  │  工具     │ ← 执行后返回 ToolMessage              │
│  └──────────┘                                       │
│    │                                                │
│    │  ToolMessage("晴天，25°C")                      │
│    ▼                                                │
│  ┌──────────┐                                       │
│  │  模型     │ ← 再次接收所有消息，生成最终回答        │
│  └──────────┘                                       │
│    │                                                │
│    │  AIMessage("今天是晴天，气温25°C，很适合出门！") │
│    ▼                                                │
│  用户看到最终回答                                     │
└────────────────────────────────────────────────────┘
```

**关键理解：** 模型本身是"无记忆"的。它每次被调用时，都需要收到**完整的消息列表**（从第一条到最新一条）才能理解上下文。你看到的"对话"其实是你的程序维护的一个不断增长的消息列表

### 1.3 三种传入消息的方式

LangChain 提供了三种等价的方式来向模型传入消息：

#### 方式一：纯文本字符串——最简情况

```python
# 当你只有一句话要问，不需要对话历史时
response = model.invoke("写一首关于春天的俳句")
```

这是最简单的形式。LangChain 内部会自动把它转换成一条 `HumanMessage`。**适用场景：** 单次、独立的问题，不需要上下文

#### 方式二：Message 对象列表——最规范

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("你是一位诗歌专家"),           # 系统指令
    HumanMessage("写一首关于春天的俳句"),         # 用户的第一条消息
    AIMessage("樱花绽放时..."),                  # 模型之前的回复（历史）
    HumanMessage("再写一首关于秋天的"),           # 用户的最新消息
]
response = model.invoke(messages)
```

每条消息都是一个明确的对象，类型安全，IDE 能提供自动补全。**适用场景：** 多轮对话、需要类型检查、生产环境

#### 方式三：字典格式——兼容 OpenAI 格式

```python
messages = [
    {"role": "system", "content": "你是一位诗歌专家"},
    {"role": "user", "content": "写一首关于春天的俳句"},
    {"role": "assistant", "content": "樱花绽放时..."},
    {"role": "user", "content": "再写一首关于秋天的"},
]
response = model.invoke(messages)
```

这种格式与 OpenAI 的 Chat Completions API 格式完全一致。**适用场景：** 从 OpenAI SDK 迁移、JSON 数据来源

**三种方式的关系：**

```css
纯文本 "你好"
   ↓ LangChain 自动转换
HumanMessage("你好")
   ↕ 完全等价
{"role": "user", "content": "你好"}
```

三种方式在功能上是等价的，可以根据偏好和场景自由选择。但当需要处理多模态内容（图片、音频）或添加元数据时，方式二和方式三更灵活

---

## 第二章：四种消息类型——谁在说话？

消息的"角色"决定了模型如何理解这条消息。LangChain 有四种核心消息类型

### 2.1 SystemMessage——"幕后导演"

```python
from langchain.messages import SystemMessage

system_msg = SystemMessage("你是一位资深 Python 开发者，擅长 Web 框架。")
```

**SystemMessage 是什么？**

它是给模型的"幕后指令"——定义模型的角色、行为准则和回答风格。用户看不到这条消息，但模型会始终遵循它。**类比：** 就像导演在演出前给演员的指示——"你演的是一个严肃的医生，说话要专业、简洁。"观众（用户）不会听到导演的话，但演员的表演会受到影响

**简单用法 vs 详细用法：**

```python
# 简单：一句话设定角色
system_msg = SystemMessage("你是一个有帮助的助手。")

# 详细：多段落指令（生产环境推荐）
system_msg = SystemMessage("""
你是一位资深 Python 开发者，擅长 Web 框架。
请遵循以下规则：
1. 每次回答都包含代码示例
2. 解释要简洁但完整
3. 使用中文回答
""")
```

**使用建议：**
- 一个对话中通常只有**一条** SystemMessage，放在消息列表的**最前面**
- 提示词越具体，模型的行为越可预测
- 有些提供商对 SystemMessage 的处理方式不同，但 LangChain 会帮你统一处理

### 2.2 HumanMessage——"用户的声音"

```python
from langchain.messages import HumanMessage

# 最简单的文本消息
human_msg = HumanMessage("什么是机器学习？")
```

**HumanMessage 是什么？**

代表用户（人类）说的话。这是最常见的消息类型——每次用户发送一条消息，就产生一个 HumanMessage

**添加元数据：**

```python
human_msg = HumanMessage(
    content="你好！",
    name="alice",     # 可选：标识不同的用户
    id="msg_123",     # 可选：唯一标识符，用于追踪
)
```

`name` 字段可以帮助模型区分不同的用户（在多用户场景中），但不是所有提供商都支持这个字段

**快捷方式：**

```python
# 这两种写法完全等价
response = model.invoke("什么是机器学习？")
response = model.invoke([HumanMessage("什么是机器学习？")])
```

当你只传一个字符串时，LangChain 自动把它包装成 `[HumanMessage(字符串)]`

### 2.3 AIMessage——"模型的回答"

```python
from langchain.messages import AIMessage

# 模型调用后自动返回 AIMessage
response = model.invoke("解释一下 AI")
print(type(response))  # <class 'langchain.messages.AIMessage'>
```

**AIMessage 是什么？**

代表模型生成的回答。你不需要手动创建它——每次调用 `model.invoke()` 后，返回的就是一个 AIMessage

**但有时候你需要手动创建 AIMessage。** 什么时候？当你要构造对话历史时：

```python
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# 手动构建一段"假装的"对话历史
# 让模型以为之前已经发生了这些对话
messages = [
    SystemMessage("你是一个有帮助的助手"),
    HumanMessage("你能帮我吗？"),
    AIMessage("当然可以！有什么需要帮忙的？"),  # ← 手动创建的 AIMessage
    HumanMessage("2+2 等于多少？"),              # 用户最新的问题
]

response = model.invoke(messages)
```

**为什么要这样做？** 因为你可能想：
- 预设一些"示范对话"让模型学习你期望的回答风格
- 恢复之前保存的对话历史
- 引导模型进入特定的对话状态

#### AIMessage 的重要属性

AIMessage 不仅仅包含文本，它还携带了丰富的元数据：

```python
response = model.invoke("你好！")

# ① 文本内容
print(response.text)           # "你好！很高兴见到你。"
print(response.content)        # 原始内容（可能是字符串或列表）

# ② 内容块（标准化格式）
print(response.content_blocks) # [{"type": "text", "text": "你好！..."}]

# ③ 工具调用（如果模型请求了工具调用）
print(response.tool_calls)     # [{"name": "get_weather", "args": {...}, "id": "..."}]

# ④ Token 使用量
print(response.usage_metadata)
# {
#     "input_tokens": 8,        # 输入消耗的 token
#     "output_tokens": 304,     # 输出消耗的 token
#     "total_tokens": 312,      # 总 token
#     "input_token_details": {"cache_read": 0},
#     "output_token_details": {"reasoning": 256}
# }

# ⑤ 响应元数据
print(response.response_metadata)  # 提供商返回的原始元数据

# ⑥ 消息 ID
print(response.id)             # 唯一标识符
```

**`text` vs `content` vs `content_blocks` 的区别——这是新手最容易混淆的地方：**

```css
response.text
  → 纯文本字符串
  → 最简单直接，只包含文本部分
  → 适合：你只需要文本回答的场景

response.content
  → 可能是字符串，也可能是列表
  → "原始格式"，保留了提供商返回的原始结构
  → 当模型返回多模态内容（如图片+文字）时，这里是一个列表

response.content_blocks
  → 始终是一个标准化的列表
  → 不同提供商的格式被统一成了一致的结构
  → 适合：需要处理不同类型内容块的场景
```

**举个例子理解它们的差异：**

```python
# 简单文本回答
response = model.invoke("你好")

response.text            # "你好！很高兴见到你。"
response.content         # "你好！很高兴见到你。"  （字符串）
response.content_blocks  # [{"type": "text", "text": "你好！很高兴见到你。"}]

# 带推理的回答（高端模型）
response = reasoning_model.invoke("解释量子力学")

response.text            # "量子力学是..."  （只有最终文本）
response.content         # [{"type": "thinking", "thinking": "用户问的是..."},
                         #  {"type": "text", "text": "量子力学是..."}]
                         # （提供商原始格式）
response.content_blocks  # [{"type": "reasoning", "reasoning": "用户问的是..."},
                         #  {"type": "text", "text": "量子力学是..."}]
                         # （LangChain 标准化格式——注意 "thinking" 变成了 "reasoning"）
```

**这就是 `content_blocks` 的价值：** 不管你用 Anthropic（返回 `"thinking"` 类型）还是 OpenAI（返回 `"reasoning"` 类型），`content_blocks` 都会统一成 `"reasoning"` 类型。你的代码不需要为每个提供商写不同的解析逻辑

#### Token 使用量——为什么要关注？

```python
response = model.invoke("你好！")
print(response.usage_metadata)
# {
#     "input_tokens": 8,
#     "output_tokens": 10,
#     "total_tokens": 18,
#     "input_token_details": {"cache_read": 0},
#     "output_token_details": {"reasoning": 0}
# }
```

| 字段 | 含义 | 为什么重要 |
|------|------|-----------|
| `input_tokens` | 你发送给模型的 token 数 | 影响请求成本 |
| `output_tokens` | 模型生成的 token 数 | 影响响应成本 |
| `total_tokens` | 总消耗 | 直接对应 API 费用 |
| `cache_read` | 从缓存读取的 token | 缓存命中可以省钱 |
| `reasoning` | 用于推理的输出 token | 推理 token 有些提供商单独计费 |

缓存命中指：这次请求里有一部分输入 token 不用重新完整计算了，而是直接从缓存里复用

#### AIMessageChunk——流式输出的碎片

在流式输出（`model.stream()`）中，你收到的不是完整的 `AIMessage`，而是一系列 `AIMessageChunk`——每个是完整消息的一小段：

```python
chunks = []        # 保存所有碎片
full_message = None  # 累加器

for chunk in model.stream("你好"):
    chunks.append(chunk)
    print(chunk.text, end="")  # 逐字打印

    # 把碎片累加成完整消息
    full_message = chunk if full_message is None else full_message + chunk

# 循环结束后，full_message 就是一个完整的 AIMessage
# 可以像正常的 AIMessage 一样使用
print(full_message.content_blocks)
```

**`+` 操作符的作用：** LangChain 重载了 `AIMessageChunk` 的 `+` 运算符，让你可以把碎片"拼接"在一起。拼完后的对象就像一个正常的 `AIMessage`，可以放入对话历史

### 2.4 ToolMessage——"工具的回执"

```python
from langchain.messages import ToolMessage

tool_message = ToolMessage(
    content="晴天，72°F",         # 工具执行的结果（文本）
    tool_call_id="call_123",     # 必须匹配 AIMessage 中工具调用的 ID
    name="get_weather",          # 工具名称
)
```

**ToolMessage 是什么？**

当模型请求调用工具后，工具执行完毕，需要把结果"告诉"模型。ToolMessage 就是这个"回执"

**完整的工具调用对话流程：**

```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage

# 第 1 步：用户提问
user_msg = HumanMessage("旧金山的天气怎么样？")

# 第 2 步：模型返回工具调用请求（而不是直接回答）
ai_msg = AIMessage(
    content=[],  # 没有文本内容
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# 第 3 步：你执行工具，把结果包装成 ToolMessage
weather_result = "晴天，72°F"
tool_msg = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # ← 必须与 ai_msg 中的 id 匹配！
)

# 第 4 步：把所有消息传给模型，让它生成最终回答
messages = [user_msg, ai_msg, tool_msg]
final_response = model.invoke(messages)
# → AIMessage("旧金山目前是晴天，气温72°F（约22°C），很适合户外活动！")
```

**`tool_call_id` 为什么必须匹配？**

模型可能在一次回复中同时请求多个工具调用，每个调用都有唯一的 `id`。当结果返回时，模型通过 `tool_call_id` 来"对号入座"——知道哪个结果对应哪个调用

```css
模型一次请求了两个工具调用：
  call_aaa: get_weather("Tokyo")
  call_bbb: get_weather("Paris")

你返回两个 ToolMessage：
  ToolMessage(content="晴天", tool_call_id="call_aaa")  → 东京的结果
  ToolMessage(content="雨天", tool_call_id="call_bbb")  → 巴黎的结果

模型通过 id 匹配：
  "call_aaa 是东京 → 晴天"
  "call_bbb 是巴黎 → 雨天"
```

#### artifact 属性——"给程序看的附加信息"

ToolMessage 有一个特殊属性 `artifact`，用于存储**不发送给模型，但程序可以使用**的附加数据：

```python
tool_msg = ToolMessage(
    content="这是最好的时代，也是最坏的时代。",  # ← 这段文字会发给模型
    tool_call_id="call_123",
    name="search_books",
    artifact={                                  # ← 这些数据不会发给模型
        "document_id": "doc_123",               #    但你的程序可以访问
        "page": 0,
    },
)

# 模型只看到 content 中的文字
# 你的程序可以用 artifact 做其他事
print(tool_msg.artifact["document_id"])  # "doc_123"
```

**为什么需要 artifact？**

你可能想把一些元数据（如文档 ID、页码、置信度分数）传递给下游程序，但不想让模型看到这些技术细节（会干扰它的回答质量、浪费 token）。`artifact` 就是这个"给程序看的附加信息通道"

### 2.5 四种消息类型总结

```css
┌─────────────────────────────────────────────────────────────┐
│                     消息列表（对话历史）                      │
│                                                             │
│  ┌─ SystemMessage ──────────────────────────────────────┐   │
│  │  "你是一位天气预报专家"                                │   │
│  │  角色：system │ 谁看到：只有模型 │ 通常：1 条           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ HumanMessage ──────────────────────────────────────┐   │
│  │  "今天天气怎么样？"                                    │   │
│  │  角色：user │ 来源：用户输入 │ 可包含：文字/图片/音频   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ AIMessage ─────────────────────────────────────────┐   │
│  │  tool_calls: [{name: "get_weather", ...}]            │   │
│  │  角色：assistant │ 来源：模型返回 │ 可包含：文字/工具调用│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ ToolMessage ───────────────────────────────────────┐   │
│  │  "晴天，25°C"                                        │   │
│  │  角色：tool │ 来源：工具执行结果 │ 必须有 tool_call_id │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ AIMessage ─────────────────────────────────────────┐   │
│  │  "今天是晴天，气温25°C，很适合出门！"                  │   │
│  │  最终回答                                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

| 消息类型 | 角色 | 谁创建的 | 主要用途 |
|---------|------|---------|---------|
| `SystemMessage` | system | 你（开发者） | 定义模型的角色和行为准则 |
| `HumanMessage` | user | 用户 | 传达用户的问题或指令 |
| `AIMessage` | assistant | 模型 | 模型的回答或工具调用请求 |
| `ToolMessage` | tool | 工具执行后 | 把工具结果返回给模型 |

---

## 第三章：消息内容（Message Content）——纸条上能写什么？

### 3.1 content 属性的灵活性

消息的 `content` 属性不仅能存文字——它是一个**灵活的容器**，可以承载文本、图片、音频、视频、文件等各种数据`content` 有两种形态：

```python
# 形态一：简单字符串（只有文字）
msg = HumanMessage(content="你好")
# content = "你好"

# 形态二：内容块列表（有多种类型的数据）
msg = HumanMessage(content=[
    {"type": "text", "text": "这张图片是什么？"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
])
# content = [{"type": "text", ...}, {"type": "image_url", ...}]
```

### 3.2 content_blocks——跨提供商的标准化

不同的模型提供商对内容的格式要求不同。比如传图片：

```
OpenAI 格式:
  {"type": "image_url", "image_url": {"url": "..."}}

Anthropic 格式:
  {"type": "image", "source": {"type": "url", "url": "..."}}

Google 格式:
  又是另一种格式...
```

**这对开发者来说是噩梦。** 每换一个提供商，你就得改一遍消息格式。LangChain 的 `content_blocks` 解决了这个问题——它定义了一套**统一的标准格式**，不管底层用的是哪个提供商，你都可以用同样的方式构建和读取消息内容：

```python
# 使用标准化的 content_blocks 构建消息
# 不管底层是 OpenAI、Anthropic 还是 Google，格式都一样
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "这张图片是什么？"},
    {"type": "image", "url": "https://example.com/cat.jpg"},
])

# LangChain 会自动把标准格式转换成提供商需要的格式
response = model.invoke([msg])  # 内部自动适配
```

**读取时也是标准化的：**

```python
# 不管模型是 Anthropic 还是 OpenAI
# content_blocks 的格式都是一致的
for block in response.content_blocks:
    if block["type"] == "text":
        print(f"文字: {block['text']}")
    elif block["type"] == "reasoning":
        print(f"推理: {block['reasoning']}")
    elif block["type"] == "image":
        print(f"图片: {block['base64'][:20]}...")
```

**`content` vs `content_blocks` 的关系：**

```css
content:
  → 提供商的原始格式
  → 写入时用这个（或者用 content_blocks，LangChain 会转换）
  → 读取时如果你确定只用一个提供商，可以用这个

content_blocks:
  → LangChain 的标准化格式
  → 写入时用这个更安全（自动适配提供商）
  → 读取时推荐用这个（跨提供商一致）
  → 是 content 的"懒解析"版本——只在你访问时才转换
```

### 3.3 标准化内容块的完整类型

LangChain 定义了以下标准内容块类型。按用途分组讲解：

#### 核心类型

**TextContentBlock——文本**

最基本的内容块，承载纯文本。

```python
{
    "type": "text",
    "text": "你好，世界！",
    "annotations": []  # 可选：文本注释（如引用来源的链接）
}
```

`annotations` 字段在服务端工具调用（如网页搜索）返回结果时特别有用——模型的回答中可能包含引用标注

**ReasoningContentBlock——推理过程**

展示模型的思考步骤。

```python
{
    "type": "reasoning",
    "reasoning": "用户问的是进化生物学的问题...",
    "extras": {"signature": "abc123"}  # 可选：提供商特有的附加数据
}
```

**跨提供商统一的威力：**

```
Anthropic 原始格式:
  {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."}
                  ↓ content_blocks 自动转换
标准化格式:
  {"type": "reasoning", "reasoning": "...", "extras": {"signature": "WaUjzkyp..."}}

OpenAI 原始格式:
  {"type": "reasoning", "summary": [{"type": "summary_text", "text": "..."}]}
                  ↓ content_blocks 自动转换
标准化格式:
  {"type": "reasoning", "reasoning": "..."}
```

你只需要检查 `block["type"] == "reasoning"`，不需要关心底层提供商是谁

#### 多模态类型

**ImageContentBlock——图片**

```python
# 方式一：通过 URL
{"type": "image", "url": "https://example.com/photo.jpg"}

# 方式二：通过 Base64 编码（本地文件）
{
    "type": "image",
    "base64": "AAAAIGZ0eXBtcDQy...",  # Base64 编码的图片数据
    "mime_type": "image/jpeg"           # 图片格式，Base64 时必填
}

# 方式三：通过提供商的文件 ID
{"type": "image", "file_id": "file-abc123"}
```

**AudioContentBlock——音频**

```python
{
    "type": "audio",
    "base64": "AAAAIGZ0eXBtcDQy...",
    "mime_type": "audio/wav"
}
```

**VideoContentBlock——视频**

```python
{
    "type": "video",
    "base64": "AAAAIGZ0eXBtcDQy...",
    "mime_type": "video/mp4"
}
```

**FileContentBlock——文件（PDF 等）**

```python
# 通过 URL
{"type": "file", "url": "https://example.com/document.pdf"}

# 通过 Base64
{
    "type": "file",
    "base64": "AAAAIGZ0eXBtcDQy...",
    "mime_type": "application/pdf"
}
```

**PlainTextContentBlock——文档文本（.txt、.md 等）**

```python
{
    "type": "text-plain",
    "text": "# Markdown 标题\n\n这是正文...",
    "mime_type": "text/markdown"
}
```

#### 三种数据来源的选择

所有多模态内容块都支持三种数据来源，什么时候用哪种？

| 来源 | 字段 | 适用场景 |
|------|------|---------|
| URL | `url` | 图片/文件有公开的网络地址 |
| Base64 | `base64` + `mime_type` | 本地文件、从数据库读取的二进制数据 |
| 文件 ID | `file_id` | 使用提供商的文件管理服务上传过的文件 |

#### 工具调用相关类型

**ToolCall——工具调用请求**

```python
{
    "type": "tool_call",
    "name": "search",              # 工具名
    "args": {"query": "天气"},      # 参数
    "id": "call_123"               # 唯一 ID
}
```

**ToolCallChunk——流式工具调用片段**

在 `model.stream()` 中，工具调用信息不是一次性到达的，而是分批到达。每个 chunk 只包含一小段：

```python
# 第一个 chunk
{"type": "tool_call_chunk", "name": "search", "id": "call_123"}
# 第二个 chunk
{"type": "tool_call_chunk", "args": "{\"qu"}
# 第三个 chunk
{"type": "tool_call_chunk", "args": "ery\":"}
# 第四个 chunk
{"type": "tool_call_chunk", "args": " \"天气\"}"}
```

注意 `args` 是逐步到达的 JSON 字符串片段，不是完整的 JSON。你需要累加所有 chunk 才能得到完整的工具调用信息

**InvalidToolCall——失败的工具调用**

当模型生成的工具调用参数不合法时（比如 JSON 格式错误），LangChain 会生成这个类型，而不是直接报错：

```python
{
    "type": "invalid_tool_call",
    "name": "search",
    "args": "这不是合法的JSON...",  # 错误的参数
    "error": "JSON parse error"     # 错误描述
}
```

这让你的代码可以优雅地处理模型的"失误"，而不是直接崩溃

#### 服务端工具调用类型

这些类型出现在模型提供商的服务端工具调用中（比如 OpenAI 的内置网页搜索）。与普通工具调用不同，服务端工具的执行发生在提供商的服务器上，你不需要自己执行。

**ServerToolCall + ServerToolResult**

```python
# 模型在服务端调用了网页搜索
{
    "type": "server_tool_call",
    "name": "web_search",
    "args": {"query": "今天的新闻"},
    "id": "ws_abc123"
}
# 服务端工具执行结果
{
    "type": "server_tool_result",
    "tool_call_id": "ws_abc123",
    "status": "success"  # 或 "error"
}
```

这些内容块直接出现在 AIMessage 的 `content_blocks` 中——整个调用和执行都在一次 API 请求中完成，你不需要像普通工具调用那样手动执行和传回结果。

---

## 第四章：多模态消息实战——发送图片、PDF、音频

### 4.1 发送图片给模型

```python
# 让模型描述一张图片
response = model.invoke([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述这张图片的内容。"},
            {"type": "image", "url": "https://example.com/sunset.jpg"},
        ]
    }
])
print(response.text)
# "这是一张美丽的日落照片，天空呈现橙色和紫色..."
```

**使用本地图片（需要 Base64 编码）：**

```python
import base64

# 读取本地图片文件并转为 Base64
with open("photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = model.invoke([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图片是什么？"},
            {
                "type": "image",
                "base64": image_data,
                "mime_type": "image/jpeg",  # Base64 时必须指定格式
            },
        ]
    }
])
```

### 4.2 发送 PDF 文档给模型

```python
response = model.invoke([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请总结这份文档的主要内容。"},
            {"type": "file", "url": "https://example.com/report.pdf"},
        ]
    }
])
```

### 4.3 发送音频给模型

```python
import base64

with open("recording.wav", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode("utf-8")

response = model.invoke([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请转录这段音频的内容。"},
            {
                "type": "audio",
                "base64": audio_data,
                "mime_type": "audio/wav",
            },
        ]
    }
])
```

**重要提醒：** 不是所有模型都支持所有类型的多模态输入。发送前请确认你使用的模型支持相应的数据类型。你可以通过 `model.profile` 查看模型的能力。

---

## 第五章：消息与对话管理——把纸条串成对话

### 5.1 为什么需要理解对话管理？

模型是**无状态**的——每次调用 `model.invoke()` 都是独立的。模型不记得你上一次问了什么。那模型怎么实现"连续对话"的？答案是：**你负责维护完整的消息列表，每次调用时把完整历史发给模型**

```python
# 第一轮对话
messages = [
    SystemMessage("你是一个有帮助的助手"),
    HumanMessage("我叫小明"),
]
response1 = model.invoke(messages)
# → AIMessage("你好，小明！很高兴认识你。")

# 第二轮对话——你需要手动把之前的消息和回复都带上
messages.append(response1)              # 把模型的回复加到历史中
messages.append(HumanMessage("我叫什么名字？"))  # 加上新问题

response2 = model.invoke(messages)
# → AIMessage("你叫小明呀！你刚刚告诉我的。")

# 如果你不带历史：
response3 = model.invoke([HumanMessage("我叫什么名字？")])
# → AIMessage("抱歉，我不知道你的名字。你能告诉我吗？")  ← 失忆了！
```

**这就是为什么 Agent 的 `checkpointer`（记忆组件）如此重要——它帮你自动维护和恢复消息历史，省去了手动管理的麻烦**

### 5.2 对话历史的增长问题

随着对话继续，消息列表会越来越长。这带来两个问题：

1. **超出上下文窗口：** 模型能处理的 token 数有上限（比如 128K），消息太多就塞不下了
2. **成本增加：** API 按输入 token 计费，每轮对话都发送完整历史，费用会越来越高

**解决方案：** LangChain 提供了消息裁剪和摘要等策略。这些在"短期记忆"相关文档中有详细介绍，这里只需要理解问题所在

### 5.3 手动构建对话历史的实际场景

**场景一：Few-shot 示范**

通过预设几轮"示范对话"来教模型你期望的回答格式：

```python
messages = [
    SystemMessage("你是一个 JSON 生成助手。用户描述一个物品，你输出 JSON。"),

    # 示范 1
    HumanMessage("一个红色的苹果"),
    AIMessage('{"name": "苹果", "color": "红色", "category": "水果"}'),

    # 示范 2
    HumanMessage("一辆蓝色的自行车"),
    AIMessage('{"name": "自行车", "color": "蓝色", "category": "交通工具"}'),

    # 实际问题
    HumanMessage("一只黄色的小鸭子"),
]

response = model.invoke(messages)
# 模型学会了格式，回答：
# '{"name": "小鸭子", "color": "黄色", "category": "玩具"}'
```

**场景二：恢复保存的对话**

```python
import json

# 从数据库/文件加载之前保存的对话
with open("chat_history.json") as f:
    saved_messages = json.load(f)

# 转换为 LangChain 消息对象
messages = []
for msg in saved_messages:
    if msg["role"] == "system":
        messages.append(SystemMessage(msg["content"]))
    elif msg["role"] == "user":
        messages.append(HumanMessage(msg["content"]))
    elif msg["role"] == "assistant":
        messages.append(AIMessage(msg["content"]))

# 加上新问题，继续对话
messages.append(HumanMessage("继续上次的话题..."))
response = model.invoke(messages)
```

---

## 第六章：所有概念的串联——消息在 Agent 中的完整流转

让我用一个完整的例子，追踪消息在 Agent 中的整个生命周期。

### 场景：用户问"旧金山的天气怎么样？"

```css
步骤 1：用户输入
═══════════════
消息列表: [
    SystemMessage("你是天气助手"),
    HumanMessage("旧金山的天气怎么样？")  ← 新增
]

         │
         ▼

步骤 2：模型收到消息列表，决定调用工具
═══════════════════════════════════
模型返回一个 AIMessage，content 为空，但 tool_calls 不为空

消息列表: [
    SystemMessage("你是天气助手"),
    HumanMessage("旧金山的天气怎么样？"),
    AIMessage(                                    ← 新增
        content=[],
        tool_calls=[{
            "name": "get_weather",
            "args": {"location": "San Francisco"},
            "id": "call_abc"
        }]
    )
]

此时 AIMessage 的 content_blocks:
  [{"type": "tool_call", "name": "get_weather",
    "args": {"location": "San Francisco"}, "id": "call_abc"}]

         │
         ▼

步骤 3：Agent 执行工具，结果作为 ToolMessage 加入
═══════════════════════════════════════════════
消息列表: [
    SystemMessage("你是天气助手"),
    HumanMessage("旧金山的天气怎么样？"),
    AIMessage(tool_calls=[{name: "get_weather", ...}]),
    ToolMessage(                                  ← 新增
        content="晴天，72°F",
        tool_call_id="call_abc",
        name="get_weather"
    )
]

         │
         ▼

步骤 4：模型收到完整的消息列表（包括工具结果），生成最终回答
═══════════════════════════════════════════════════════
消息列表: [
    SystemMessage("你是天气助手"),
    HumanMessage("旧金山的天气怎么样？"),
    AIMessage(tool_calls=[...]),
    ToolMessage(content="晴天，72°F", ...),
    AIMessage(                                    ← 新增
        content="旧金山目前是晴天，气温约72°F（22°C），
                 非常适合出门散步！"
    )
]

此时最后一个 AIMessage 的各属性:
  .text            → "旧金山目前是晴天..."
  .content         → "旧金山目前是晴天..."
  .content_blocks  → [{"type": "text", "text": "旧金山目前是晴天..."}]
  .tool_calls      → []（没有工具调用，因为这次是最终回答）
  .usage_metadata  → {"input_tokens": 85, "output_tokens": 32, ...}
```

**关键观察：**
- 消息列表是**只增不减**的——每一步都往列表末尾添加新消息
- 模型每次都收到**完整的列表**——它通过阅读所有历史消息来理解上下文
- 不同类型的消息交替出现：Human → AI（工具调用）→ Tool → AI（最终回答）

---

## 第七章：速查手册

### 消息类型速查

| 类型 | 角色 | 字典格式中的 role | 创建者 | 关键属性 |
|------|------|-----------------|--------|---------|
| `SystemMessage` | 系统 | `"system"` | 开发者 | `content` |
| `HumanMessage` | 用户 | `"user"` | 用户/开发者 | `content`, `name`, `id` |
| `AIMessage` | 助手 | `"assistant"` | 模型 | `text`, `content`, `content_blocks`, `tool_calls`, `usage_metadata` |
| `ToolMessage` | 工具 | `"tool"` | 工具执行后 | `content`, `tool_call_id`, `name`, `artifact` |
| `AIMessageChunk` | 助手 | — | 流式输出 | 同 AIMessage，但是碎片化的 |

### 内容块类型速查

| 类型标识 | 用途 | 关键字段 |
|---------|------|---------|
| `"text"` | 普通文本 | `text`, `annotations` |
| `"reasoning"` | 推理过程 | `reasoning`, `extras` |
| `"image"` | 图片 | `url` 或 `base64` + `mime_type` |
| `"audio"` | 音频 | `base64` + `mime_type` |
| `"video"` | 视频 | `base64` + `mime_type` |
| `"file"` | 文件（PDF等） | `url` 或 `base64` + `mime_type` |
| `"text-plain"` | 文档文本 | `text`, `mime_type` |
| `"tool_call"` | 工具调用请求 | `name`, `args`, `id` |
| `"tool_call_chunk"` | 流式工具调用片段 | `name`, `args`(部分), `id` |
| `"invalid_tool_call"` | 格式错误的工具调用 | `name`, `args`, `error` |
| `"server_tool_call"` | 服务端工具调用 | `name`, `args`, `id` |
| `"server_tool_result"` | 服务端工具结果 | `tool_call_id`, `status` |

### AIMessage 属性速查

| 属性 | 返回类型 | 说明 |
|------|---------|------|
| `.text` | `str` | 纯文本内容 |
| `.content` | `str` 或 `list` | 原始内容（提供商格式） |
| `.content_blocks` | `list[dict]` | 标准化内容块列表 |
| `.tool_calls` | `list[dict]` | 工具调用列表 |
| `.usage_metadata` | `dict` 或 `None` | Token 使用量 |
| `.response_metadata` | `dict` 或 `None` | 响应元数据 |
| `.id` | `str` | 唯一标识符 |

### 与前两节知识的关系

```
第一节（Agent）
  → 理解了 Agent 的运行循环
  → 消息是循环中每一步的"载体"

第二节（Model）
  → 理解了模型的调用方式（invoke/stream/batch）
  → 消息是模型的输入和输出格式

本节（Messages）  ← 你在这里
  → 深入理解了消息本身的结构
  → 了解了四种消息类型各自的用途
  → 掌握了多模态内容的传递方式
  → 理解了 content_blocks 的跨提供商标准化

下一步建议学习：
  → Tools（工具）——深入学习工具的定义和使用
  → Middleware（中间件）——学习如何裁剪和管理消息历史
  → Short-term Memory（短期记忆）——学习自动对话管理
```
