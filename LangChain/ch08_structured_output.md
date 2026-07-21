# LangChain 结构化输出（Structured Output）深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第七节。前六节分别学了 Agent（整体架构）、Model（大脑）、Messages（消息）、Tools（手和脚）、Short-term Memory（对话记忆）、Streaming（实时播报）。本节学习 Structured Output——让 Agent 的输出**不再是随意的自然语言**，而是**你预定义好格式的结构化数据**。如果说前面六节解决了 Agent "能做什么"和"怎么做"的问题，那结构化输出解决的是**"怎么把结果交给程序用"**的问题

---

## 第一章：为什么需要结构化输出？

### 1.1 自然语言输出的痛点

假设你让 Agent 分析一条产品评论：

```
你: "分析这条评论：'东西很好，物流快，就是贵了点，给4分'"
Agent: "这是一条正面评论。用户对产品质量满意，物流速度也很好，
       但认为价格偏高。总体评分4分（满分5分）。"
```

这段回答人类读起来很舒服，但如果你的程序需要——

- 把评分存到数据库？→ 你得从文本中抠出 "4"
- 判断情感是正面还是负面？→ 你得解析 "正面评论" 这几个字
- 提取关键词？→ 你得想办法识别 "质量满意"、"物流快"、"价格偏高"

每一步都需要写额外的文本解析代码，而且 LLM 的回答格式不固定——下次它可能说"用户打了四分"而不是"4分"，你的解析代码就挂了

### 1.2 结构化输出的解决方案

有了结构化输出，同样的请求返回的是：

```python
ProductReview(
    rating=4,
    sentiment="positive",
    key_points=["质量好", "物流快", "价格偏高"]
)
```

这是一个 **Pydantic 模型实例**——字段名固定、类型固定、可以直接用。你的程序不需要做任何解析，直接 `result.rating` 就拿到评分了

### 1.3 核心思路

```
没有结构化输出:
  用户输入 → Agent 思考 → 自然语言文本 → 你的程序需要解析文本

有结构化输出:
  用户输入 → Agent 思考 → 符合 Schema 的结构化数据 → 你的程序直接使用
                              ↑
                     你预先定义好的数据格式（Schema）
```

---

## 第二章：`create_agent` 中的 `response_format` 参数

### 2.1 基本使用方式

在 `create_agent` 中通过 `response_format` 参数指定输出格式：

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class ContactInfo(BaseModel):
    """某人的联系方式。"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="电话号码")


# 创建 Agent 时指定结构化输出格式
agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # ← 告诉 Agent：你的输出必须是 ContactInfo 格式
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系方式：张三, zhangsan@example.com, 13800138000"}]
})

# 结构化数据在 "structured_response" 键中
print(result["structured_response"])
# ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')

# 直接用字段访问
print(result["structured_response"].name)   # '张三'
print(result["structured_response"].email)  # 'zhangsan@example.com'
```

**关键要点：**

- 结构化输出通过 `result["structured_response"]` 获取，而不是从 `messages` 中解析
- LangChain 会自动验证模型输出是否符合你定义的 Schema
- 如果验证失败，还有自动重试机制（后面会讲）

### 2.2 `response_format` 可以接受什么？

`response_format` 参数有四种输入形式：

| 输入形式 | 说明 | 示例 |
|:-------:|------|:----:|
| 直接传 Schema 类型 | 自动选择最佳策略 | `response_format=ContactInfo` |
| `ProviderStrategy[T]` | 强制使用提供商原生的结构化输出 | `response_format=ProviderStrategy(ContactInfo)` |
| `ToolStrategy[T]` | 强制使用工具调用来实现结构化输出 | `response_format=ToolStrategy(ContactInfo)` |
| `None` | 不使用结构化输出（默认） | `response_format=None` |

**直接传类型时，LangChain 的自动选择逻辑：**

```
你传入: response_format=ContactInfo

LangChain 内部判断:
  ├── 模型支持原生结构化输出（如 OpenAI、Anthropic）？
  │     └── 是 → 使用 ProviderStrategy（更可靠）
  └── 模型不支持原生结构化输出？
        └── 使用 ToolStrategy（通过工具调用实现）
```

大多数情况下，直接传类型就好了。只有当你需要精确控制策略或进行细粒度配置时，才需要显式使用 `ProviderStrategy` 或 `ToolStrategy`

### 2.3 支持的 Schema 定义方式

LangChain 支持四种方式来定义输出格式：

#### 方式一：Pydantic Model（推荐）

```python
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    """某人的联系方式。"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="电话号码")
```

**优点：** 功能最强大——支持字段验证（如 `ge=1, le=5`）、默认值、复杂嵌套类型。返回的是 Pydantic 模型实例，可以直接用 `.name` 访问字段

#### 方式二：Dataclass

```python
from dataclasses import dataclass

@dataclass
class ContactInfo:
    """某人的联系方式。"""
    name: str   # 姓名
    email: str  # 邮箱地址
    phone: str  # 电话号码
```

**优点：** Python 原生语法，无需安装 Pydantic。返回的是**字典**（注意：不是 dataclass 实例）

#### 方式三：TypedDict

```python
from typing_extensions import TypedDict

class ContactInfo(TypedDict):
    """某人的联系方式。"""
    name: str   # 姓名
    email: str  # 邮箱地址
    phone: str  # 电话号码
```

**优点：** 轻量级，适合简单场景。返回的也是**字典**

#### 方式四：JSON Schema

```python
contact_info_schema = {
    "type": "object",
    "description": "某人的联系方式。",
    "properties": {
        "name": {"type": "string", "description": "姓名"},
        "email": {"type": "string", "description": "邮箱地址"},
        "phone": {"type": "string", "description": "电话号码"}
    },
    "required": ["name", "email", "phone"]
}
```

**优点：** 可以从外部配置文件加载，适合动态 Schema。返回的是**字典**

#### 四种方式的对比

| 方式 | 返回类型 | 字段验证 | 适用场景 |
|:----:|:-------:|:-------:|:-------:|
| Pydantic Model | Pydantic 实例 | 支持（最强） | 需要严格验证的生产环境 |
| Dataclass | dict | 基本类型检查 | 简单场景，不想装 Pydantic |
| TypedDict | dict | 无运行时验证 | 轻量级，快速原型 |
| JSON Schema | dict | 通过 Schema 验证 | 动态 Schema，外部配置 |

---

## 第三章：两种策略的深入理解

### 3.1 Provider Strategy（提供商原生策略）

**什么是"原生结构化输出"？**

一些模型提供商（OpenAI、Anthropic、xAI 等）在 API 层面直接支持结构化输出——你把 Schema 发给 API，API 保证返回的数据严格符合这个 Schema。这不是 LangChain 的功能，而是模型提供商自己的能力

```
你的 Schema ──→ 提供商 API ──→ 保证符合 Schema 的 JSON
                   ↑
              API 层面强制约束
              （不会偏离 Schema）
```

**使用方式：**

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-5",
    response_format=ProviderStrategy(ContactInfo)
    # 或者直接: response_format=ContactInfo（自动选择 ProviderStrategy）
)
```

**ProviderStrategy 的参数：**

```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]      # 必填：输出格式的 Schema
    strict: bool | None = None  # 可选：是否启用严格模式
```

- `schema`：你定义的输出格式（Pydantic Model / Dataclass / TypedDict / JSON Schema）
- `strict`：部分提供商（如 OpenAI、xAI）支持的严格模式。启用后，API 会更严格地遵循 Schema，但可能限制 Schema 的复杂度

**优点：**

- 可靠性最高——模型提供商在 API 层面保证输出格式
- 性能好——不需要额外的工具调用开销

**限制：**

- 不是所有模型都支持
- Schema 的复杂度可能受限（取决于提供商）

### 3.2 Tool Strategy（工具调用策略）

**原理：**

当模型不支持原生结构化输出时，LangChain 用了一个巧妙的方法——**把你的 Schema 伪装成一个工具（tool）**，让模型通过"调用这个工具"来输出结构化数据

```
传统工具调用:
  模型 → 调用 get_weather(city="北京") → 执行真正的函数 → 返回结果

结构化输出的工具调用:
  模型 → 调用 ContactInfo(name="张三", email="...") → 不执行函数
                                                        → 直接把参数当作结构化输出
```

本质上就是"借用"了工具调用的参数格式来传递结构化数据

**使用方式：**

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="gpt-5",
    tools=tools,  # 可以同时有普通工具和结构化输出
    response_format=ToolStrategy(ProductReview)
)
```

**ToolStrategy 的参数：**

```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]                  # 必填：输出格式的 Schema
    tool_message_content: str | None       # 可选：自定义工具消息内容
    handle_errors: Union[bool, str, ...]   # 可选：错误处理策略（默认 True）
```

**与 ProviderStrategy 的对比：**

| 特性 | ProviderStrategy | ToolStrategy |
|------|-----------------|-------------|
| 实现方式 | 提供商 API 原生支持 | 通过工具调用模拟 |
| 可靠性 | 最高（API 保证） | 较高（模型可能犯错，但有重试机制） |
| 兼容性 | 仅限支持的提供商 | 所有支持工具调用的模型 |
| 额外功能 | `strict` 模式 | Union 类型、自定义错误处理、自定义消息 |
| 性能开销 | 低 | 稍高（需要一次工具调用） |

---

## 第四章：ToolStrategy 的独有能力

ToolStrategy 有几个 ProviderStrategy 不具备的强大功能

### 4.1 Union 类型——让模型自己选格式

有时候，同一个输入可能对应不同的输出格式。比如分析客户反馈时，有些是**产品评论**，有些是**投诉**。你可以用 `Union` 类型让模型自己判断：

```python
from pydantic import BaseModel, Field
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductReview(BaseModel):
    """产品评论分析。"""
    rating: int | None = Field(description="评分（1-5）", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="情感倾向")
    key_points: list[str] = Field(description="关键要点")


class CustomerComplaint(BaseModel):
    """客户投诉。"""
    issue_type: Literal["product", "service", "shipping", "billing"] = Field(
        description="问题类型"
    )
    severity: Literal["low", "medium", "high"] = Field(description="严重程度")
    description: str = Field(description="问题描述")


agent = create_agent(
    model="gpt-5",
    response_format=ToolStrategy(Union[ProductReview, CustomerComplaint])
    # ← 模型会根据输入内容自动选择合适的格式
)

# 输入是评论 → 返回 ProductReview
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "分析：'东西不错，5星好评！'"}]
})
print(result1["structured_response"])
# ProductReview(rating=5, sentiment='positive', key_points=['质量好'])

# 输入是投诉 → 返回 CustomerComplaint
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "分析：'快递丢了，联系客服没人理！'"}]
})
print(result2["structured_response"])
# CustomerComplaint(issue_type='shipping', severity='high', description='快递丢失且客服无响应')
```

**工作原理：** LangChain 把 `Union[ProductReview, CustomerComplaint]` 中的每个类型都注册为一个"工具"。模型根据输入内容选择调用哪个"工具"，从而决定输出格式

### 4.2 自定义工具消息内容

当 ToolStrategy 生成结构化输出时，它会在对话历史中插入一条 `ToolMessage`。默认内容是：

```
Returning structured response: {'name': '张三', 'email': '...', ...}
```

你可以通过 `tool_message_content` 自定义这条消息：

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class MeetingAction(BaseModel):
    """从会议记录中提取的待办事项。"""
    task: str = Field(description="具体任务")
    assignee: str = Field(description="负责人")
    priority: Literal["low", "medium", "high"] = Field(description="优先级")


agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="待办事项已记录！"  # ← 自定义消息
    )
)
```

**对话历史中的体现：**

```
自定义消息:
  ToolMessage: "待办事项已记录！"

默认消息:
  ToolMessage: "Returning structured response: {'task': '...', 'assignee': '...', ...}"
```

**为什么要自定义？** 在多轮对话中，这条 ToolMessage 会出现在对话历史里。如果你不希望把原始数据暴露在对话中（比如包含敏感信息），或者想让对话看起来更自然，就可以自定义这条消息

### 4.3 错误处理与自动重试

这是 ToolStrategy 最强大的功能之一。因为结构化输出是通过工具调用模拟的，模型有可能"犯错"——比如返回不符合 Schema 的数据。ToolStrategy 有内置的错误处理和重试机制

#### 错误类型一：模型返回了多个结构化输出

当你用 `Union` 类型时，模型可能同时调用多个"工具"（即返回多个结构化输出），但我们通常只期望一个：

```
期望:
  模型 → 调用 ContactInfo(...)  ← 只返回一个

实际（错误情况）:
  模型 → 调用 ContactInfo(...)
       → 调用 EventDetails(...)  ← 多返回了一个！
```

**LangChain 的处理方式：**

1. 检测到多个结构化输出
2. 给模型发送错误反馈（ToolMessage）
3. 模型根据反馈重新生成，这次只返回一个

```
对话流程:

  Human: "提取信息：张三(zhang@email.com) 正在组织3月15日的技术大会"

  AI: [调用 ContactInfo(...), 调用 EventDetails(...)]  ← 错误：返回了两个

  ToolMessage: "Error: Model incorrectly returned multiple structured responses
               (ContactInfo, EventDetails) when only one is expected."

  AI: [调用 ContactInfo(...)]  ← 重试：这次只返回一个 ✓
```

#### 错误类型二：数据不符合 Schema 验证

模型返回的数据可能违反你定义的约束条件（如评分范围）：

```
Schema 定义: rating: int = Field(ge=1, le=5)  ← 评分必须 1-5

模型返回: rating=10  ← 违反约束！
```

**LangChain 的处理方式：**

```
对话流程:

  Human: "分析评论：'太棒了，满分10分！'"

  AI: [调用 ProductRating(rating=10, comment="太棒了")]  ← 错误：rating > 5

  ToolMessage: "Error: Failed to parse structured output: 1 validation error
               for ProductRating.rating - Input should be less than or equal
               to 5. Please fix your mistakes."

  AI: [调用 ProductRating(rating=5, comment="太棒了")]  ← 重试：修正为 5 ✓
```

#### `handle_errors` 参数详解

`handle_errors` 控制错误处理的策略，有六种用法：

```python
# 1. True（默认）—— 捕获所有错误，使用默认错误消息重试
ToolStrategy(schema=MySchema, handle_errors=True)

# 2. 字符串 —— 捕获所有错误，使用你指定的固定消息重试
ToolStrategy(schema=MySchema, handle_errors="请提供1-5的评分并附上评论。")

# 3. 异常类型 —— 只捕获特定类型的错误，其他错误直接抛出
ToolStrategy(schema=MySchema, handle_errors=ValueError)

# 4. 异常类型元组 —— 只捕获指定的多种错误类型
ToolStrategy(schema=MySchema, handle_errors=(ValueError, TypeError))

# 5. 自定义函数 —— 用函数处理错误，返回自定义的错误消息
def my_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "格式有误，请重试。"
    return f"发生错误: {str(error)}"

ToolStrategy(schema=MySchema, handle_errors=my_handler)

# 6. False —— 不处理错误，直接抛出异常（不重试）
ToolStrategy(schema=MySchema, handle_errors=False)
```

**六种策略的选择指南：**

```
你需要最大容错性？
  → handle_errors=True（默认，推荐大多数场景）

你想给模型更明确的修正指引？
  → handle_errors="你的自定义提示"

你想对不同错误类型做不同处理？
  → handle_errors=自定义函数

你只关心某种特定错误？
  → handle_errors=ValueError 或 (ValueError, TypeError)

你在调试，想看到原始错误？
  → handle_errors=False
```

**自定义错误处理函数的完整示例：**

```python
from langchain.agents.structured_output import (
    StructuredOutputValidationError,
    MultipleStructuredOutputsError,
)


def custom_error_handler(error: Exception) -> str:
    """根据错误类型返回不同的错误提示。"""
    if isinstance(error, StructuredOutputValidationError):
        # Schema 验证失败（如字段类型错误、约束违反等）
        return "输出格式有误，请检查字段类型和取值范围后重试。"
    elif isinstance(error, MultipleStructuredOutputsError):
        # 返回了多个结构化输出
        return "请只返回一个最相关的结构化输出。"
    else:
        # 其他未知错误
        return f"发生错误: {str(error)}"


agent = create_agent(
    model="gpt-5",
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails],
        handle_errors=custom_error_handler
    )
)
```

---

## 第五章：结构化输出与工具的关系

### 5.1 结构化输出和工具可以共存

一个 Agent 可以同时拥有**普通工具**和**结构化输出**：

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent


def search_reviews(product_id: str) -> str:
    """搜索产品评论。"""
    return "用户说：东西很好，物流快，就是贵了点"


class ReviewAnalysis(BaseModel):
    """评论分析结果。"""
    sentiment: Literal["positive", "negative", "neutral"]
    key_points: list[str]
    overall_score: int = Field(ge=1, le=10)


agent = create_agent(
    model="gpt-5",
    tools=[search_reviews],             # ← 普通工具
    response_format=ReviewAnalysis       # ← 结构化输出
)
```

**执行流程：**

```
用户: "分析产品 A001 的评论"

第一轮 (ReAct):
  模型 → 调用 search_reviews(product_id="A001") → 获取评论文本
            ↑ 这是普通工具调用

第二轮 (ReAct):
  模型 → 调用 ReviewAnalysis(sentiment="positive", ...) → 结构化输出
            ↑ 这是结构化输出的"伪工具调用"

结果:
  result["structured_response"] = ReviewAnalysis(...)
```

**关键理解：** 模型会先用普通工具收集信息，最后用结构化输出的"伪工具"生成格式化结果。从模型的视角看，结构化输出就是"另一个工具"——只不过这个工具不会真正执行函数，而是把参数直接作为输出。

### 5.2 使用结构化输出时的模型要求

如果同时指定了 `tools` 和 `response_format`，模型必须支持**同时使用工具和结构化输出**。在 ToolStrategy 下，这意味着模型需要能在多个工具中正确选择；在 ProviderStrategy 下，模型需要同时处理原生结构化输出和工具调用。

---

## 第六章：策略自动选择的内部机制

### 6.1 LangChain 如何决定用哪个策略？

当你直接传 Schema 类型时（如 `response_format=ContactInfo`），LangChain 的选择逻辑如下：

```python
# 伪代码——LangChain 内部逻辑
def select_strategy(model, schema):
    # 1. 检查模型的 profile 数据
    profile = model.get_profile()

    if profile.get("structured_output") is True:
        # 模型支持原生结构化输出 → 用 ProviderStrategy
        return ProviderStrategy(schema)
    else:
        # 不支持 → 用 ToolStrategy
        return ToolStrategy(schema)
```

### 6.2 模型的 Profile 数据

LangChain >= 1.1 开始，模型的能力信息是从 **profile 数据**中动态读取的（而不是硬编码）。如果某个模型的 profile 数据不完整或不可用，你可以手动指定：

```python
from langchain.chat_models import init_chat_model

# 手动指定模型能力
custom_profile = {
    "structured_output": True,  # 告诉 LangChain 这个模型支持原生结构化输出
    # ... 其他能力配置
}

model = init_chat_model("some-model", profile=custom_profile)

agent = create_agent(
    model=model,
    response_format=ContactInfo  # 会使用 ProviderStrategy
)
```

### 6.3 手动指定策略 vs 自动选择

| 场景 | 建议 |
|------|------|
| 日常使用，主流模型 | 直接传类型：`response_format=MySchema` |
| 需要严格模式 | 显式使用：`ProviderStrategy(MySchema, strict=True)` |
| 需要 Union 类型 | 显式使用：`ToolStrategy(Union[A, B])` |
| 需要自定义错误处理 | 显式使用：`ToolStrategy(MySchema, handle_errors=...)` |
| 模型 profile 不准确 | 显式指定策略，避免自动选择出错 |

---

## 第七章：`structured_response` 在 Agent 状态中的位置

### 7.1 它和 messages 的关系

结构化输出**不替代** messages——它是 Agent 状态中的一个**额外字段**：

```python
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

# Agent 状态中的主要字段：
result["messages"]              # 对话历史（包括所有 Human/AI/Tool 消息）
result["structured_response"]   # 结构化输出（你定义的 Schema 实例或字典）
```

**对话历史中仍然包含完整的消息流：**

```
result["messages"] 的内容:

  HumanMessage: "分析这条评论：'东西很好，5星！'"
  AIMessage:     [tool_call: ProductReview(rating=5, sentiment="positive", ...)]
  ToolMessage:   "Returning structured response: {...}"
     ↑
     这就是 tool_message_content 控制的内容
```

### 7.2 与 v2 格式的配合

结合第六节学的 v2 流式格式：

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    version="v2",
)

# v2 格式的返回值
result.value["structured_response"]  # 结构化输出
result.value["messages"]             # 对话历史
result.interrupts                    # 中断信息
```

---

## 第八章：概念串联——结构化输出在 Agent 生态中的位置

### 8.1 与前几节知识的关联

```
第一节（Agent）
  → create_agent 的 response_format 参数是本节的入口
  → ReAct 循环中，结构化输出是最后一步（模型生成结构化数据 → 循环结束）

第二节（Model）
  → ProviderStrategy 依赖模型的原生结构化输出能力
  → model.with_structured_output() 是更底层的 API（本节讲的是 Agent 层面的封装）

第三节（Messages）
  → ToolStrategy 生成的结构化输出会体现为 AIMessage 中的 tool_call
  → 错误重试时的 ToolMessage 遵循第三节学的消息格式

第四节（Tools）
  → ToolStrategy 本质上是"把 Schema 当作工具"
  → 结构化输出的"伪工具"和普通工具共存于 Agent 的工具列表中
  → 模型需要区分"该调用普通工具"还是"该输出结构化数据"

第五节（Short-term Memory）
  → 结构化输出的对话历史（包括错误重试）会被保存在 memory 中
  → 多轮对话中 tool_message_content 影响对话上下文质量

第六节（Streaming）
  → 结构化输出同样可以流式传输
  → 通过 messages 模式可以看到结构化输出的 tool_call 逐步构造
  → 通过 updates 模式可以获取完成的结构化结果

本节（Structured Output）← 你在这里
  → 两种策略: ProviderStrategy（原生）、ToolStrategy（工具模拟）
  → 四种 Schema 定义方式: Pydantic / Dataclass / TypedDict / JSON Schema
  → ToolStrategy 独有功能: Union 类型、错误处理、自定义消息
  → 结果在 result["structured_response"] 中获取
```

### 8.2 什么时候该用结构化输出？

```
需要结构化输出的场景:
  ✓ 后续程序需要直接使用 Agent 的输出（如存数据库、调 API）
  ✓ 输出格式有明确的 Schema（字段名、类型、约束都已知）
  ✓ 需要保证输出一致性（同样的输入总是产生相同结构的输出）
  ✓ 需要类型安全和验证（如评分必须在 1-5 之间）

不需要结构化输出的场景:
  ✗ Agent 只是在和人聊天（自然语言回复就够了）
  ✗ 输出格式不固定（如"帮我写篇文章"）
  ✗ 简单的问答（"今天天气怎么样？"）
```

---

## 第九章：速查手册

### 最简用法

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class MyOutput(BaseModel):
    """输出描述。"""
    field1: str = Field(description="字段1描述")
    field2: int = Field(description="字段2描述")

agent = create_agent(model="gpt-5", response_format=MyOutput)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
output = result["structured_response"]  # MyOutput 实例
```

### 策略选择速查

| 你的需求 | 用什么 |
|---------|-------|
| 简单场景，自动选择 | `response_format=MySchema` |
| 强制原生结构化输出 | `response_format=ProviderStrategy(MySchema)` |
| 启用严格模式 | `response_format=ProviderStrategy(MySchema, strict=True)` |
| 强制工具调用策略 | `response_format=ToolStrategy(MySchema)` |
| 多种可能的输出格式 | `response_format=ToolStrategy(Union[A, B, C])` |
| 自定义错误处理 | `response_format=ToolStrategy(MySchema, handle_errors=...)` |

### Schema 定义方式速查

| 方式 | 返回类型 | 字段验证能力 |
|:----:|:-------:|:----------:|
| Pydantic BaseModel | 模型实例 | 最强（ge/le/regex/自定义验证器） |
| @dataclass | dict | 基本类型注解 |
| TypedDict | dict | 无运行时验证 |
| JSON Schema dict | dict | 通过 Schema 属性验证 |

### 错误处理速查

| `handle_errors` 值 | 行为 |
|:-----------------:|:----:|
| `True`（默认） | 捕获所有错误，默认消息重试 |
| `"自定义消息"` | 捕获所有错误，用你的消息重试 |
| `ValueError` | 只捕获 ValueError，其他抛出 |
| `(ValueError, TypeError)` | 只捕获这两种，其他抛出 |
| `callable` | 用你的函数生成错误消息 |
| `False` | 不捕获，直接抛出异常 |

### 与前几节的关系速查

```
Agent (第一节)     → response_format 是 create_agent 的参数
Model (第二节)     → ProviderStrategy 依赖模型原生能力
Messages (第三节)  → 错误重试通过 ToolMessage 传递
Tools (第四节)     → ToolStrategy 把 Schema 当工具
Memory (第五节)    → 结构化输出的对话历史可被记忆
Streaming (第六节) → 结构化输出可以流式传输

下一步建议学习:
  → Human-in-the-loop（人机协作）
  → Multi-agent（多 Agent 架构）
  → Guardrails（安全护栏）
```