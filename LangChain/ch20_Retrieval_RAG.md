# LangChain 检索（Retrieval）与 RAG 深入理解教程

> **阅读提示：** 本文是 LangChain 系列教程的第十四节。前十三节讲的都是"让 Agent 怎么思考和行动"——工具、中间件、上下文工程、多智能体。本节换一个角度：**当 Agent 需要的知识不在模型的训练数据里时，怎么办？** 这就引出了整个"检索（Retrieval）"和 **RAG（检索增强生成）**的话题。学完本节，你将理解为什么需要 RAG、RAG 的基本原理、以及三种主流的 RAG 架构各自适合什么场景

---

## 第一章：为什么需要检索？

### 1.1 LLM 的两大先天缺陷

再强大的大语言模型，都有两个无法回避的限制：

```
缺陷一：上下文有限
  → 模型一次能"看到"的文本有限（几十 K 到几百万 tokens）
  → 你不可能把整个公司的文档塞进一个 prompt
  → 即使塞得下，成本和延迟也会爆炸

缺陷二：知识静态
  → 模型的训练数据截止到某个时间点
  → 截止日之后的事情、你公司内部的资料、今天发生的新闻——它都不知道
  → 模型本身无法"学习"新知识（除非重新训练）
```

### 1.2 检索的解决思路

**检索（Retrieval）**的核心思想很朴素：

> **与其把所有知识都塞进模型，不如让模型在需要时"查资料"**

这和人类工作的方式如出一辙——一个再博学的专家也要查参考书、问同事、搜数据库。LLM 也应该如此

```
传统做法:
  用户问题 → LLM（只靠记忆） → 答案
  问题: 记忆不够、不准、不新

检索做法:
  用户问题 → 先从知识库里找相关资料 → LLM（看着资料答） → 答案
  优点: 资料新鲜、来源可追溯、回答有依据
```

这就是 **RAG（Retrieval-Augmented Generation，检索增强生成）**的全部本质——用检索来增强生成

### 1.3 RAG 解决了什么实际问题？

```
1. 私有知识访问
   → 公司内部文档、产品手册、客户资料
   → 模型训练时根本没见过这些

2. 时效性
   → 今天的新闻、本周的财报、实时股价
   → 模型截止日之后的一切

3. 可追溯性
   → 答案可以标注来源："这个数据来自 2024 年年报第 15 页"
   → 对金融、医疗、法律等高可信场景至关重要

4. 降低幻觉
   → 模型不再"凭记忆瞎编"
   → 有明确的参考资料约束
```

---

## 第二章：知识库与检索流水线

### 2.1 什么是知识库？

**知识库（Knowledge Base）**就是一个存放文档或结构化数据的仓库，供检索使用。它可以是：

```
- 一堆 PDF 文件
- 一个 Notion 工作区
- 公司的 Confluence 站点
- 一个 SQL 数据库
- 一个向量数据库
- ......
```

**一个重要提醒：** 如果你已经有知识库了（比如公司的 SQL 数据库、CRM、文档系统），**你不需要重建它**。有两个选择：

```
选择一: 把现有知识库包装成工具
  → 给 Agent 一个 "search_crm" 或 "query_database" 工具
  → Agent 自己决定什么时候查（Agentic RAG）

选择二: 查完再喂给 LLM
  → 在代码里先调用现有系统，拿到结果
  → 把结果作为 context 传给 LLM（2-Step RAG）
```

**只有当你没有现成知识库、需要从零构建一个可搜索的资料库时**，才需要用 LangChain 的文档加载器、嵌入模型、向量存储去构建一个全新的知识库

### 2.2 检索流水线（Retrieval Pipeline）

一个典型的检索工作流长这样：

```
┌─ 构建阶段（一次性，离线）──────────────────────┐
│                                                 │
│  数据源 (Google Drive, Slack, Notion, ...)      │
│       ↓                                         │
│  Document Loaders（文档加载器）                  │
│       ↓                                         │
│  Documents（标准化文档对象）                      │
│       ↓                                         │
│  Text Splitters（切分成小块）                    │
│       ↓                                         │
│  Embedding Model（转成向量）                     │
│       ↓                                         │
│  Vector Store（存入向量数据库）                   │
│                                                 │
└─────────────────────────────────────────────────┘

┌─ 查询阶段（每次查询都执行）──────────────────────┐
│                                                 │
│  User Query（用户问题）                          │
│       ↓                                         │
│  Embedding Model（把问题也转成向量）             │
│       ↓                                         │
│  Vector Store（搜索最相似的文档块）              │
│       ↓                                         │
│  Retriever 返回相关文档                          │
│       ↓                                         │
│  LLM（看着这些文档生成答案）                      │
│       ↓                                         │
│  Answer                                         │
│                                                 │
└─────────────────────────────────────────────────┘
```

这个流水线的每一步都是**模块化**的——你可以随意替换加载器、切分器、嵌入模型、向量存储，而不用重写整个应用。

### 2.3 五个核心组件

| 组件 | 作用 | 典型输入 → 输出 |
|------|------|---------------|
| **Document Loaders** | 从外部源加载数据 | 文件/API → 标准化的 `Document` 对象 |
| **Text Splitters** | 把长文档切成小块 | 长文档 → 多个短 chunk |
| **Embedding Models** | 把文本变成向量 | 文本 → 浮点数数组 |
| **Vector Stores** | 存储和搜索向量 | 向量 → 相似向量 |
| **Retrievers** | 统一的检索接口 | 查询字符串 → 相关文档列表 |

我们逐个理解它们的"为什么"

#### 2.3.1 为什么需要 Document Loaders？

因为**数据源五花八门**，PDF、Word、Markdown、网页、Notion API、Google Drive、Slack 消息……Document Loader 的作用是把这些数据源**统一转换成 LangChain 的 `Document` 对象**，让后续所有步骤都处理同一种数据结构

```python
# Document 对象的基本结构
Document(
    page_content="这是文档的实际内容...",
    metadata={
        "source": "/path/to/file.pdf",
        "page": 3,
        "author": "张三",
        # 任何你想记录的元数据
    }
)
```

**metadata 的重要性：** 它让你可以在检索时知道"这段内容从哪来"——回答用户时能给出来源引用

#### 2.3.2 为什么需要 Text Splitters？

两个原因：

```
原因一: 向量检索的单位必须合适
  → 把整本 500 页的书变成一个向量是没意义的
  → 因为这个向量会是"平均的"，什么都代表，什么都不精确
  → 必须切成小块，每块表达一个相对聚焦的意思

原因二: 上下文窗口有限
  → 检索到的文档最终要塞进 LLM 的 prompt
  → 如果每块都是整本书，一次只能塞几块就满了
  → 切小后可以塞更多、更精确的块
```

**切多大合适？** 这是一门手艺活：
```
- 太小（比如 50 字）: 丢失上下文，每块都是碎片
- 太大（比如 5000 字）: 检索不精确，浪费 token
- 经验值: 500-1500 字符一块，相邻块重叠 10-20%
```

**重叠（overlap）是什么？** 切块时故意让相邻块有部分重复内容——这样即使切点不巧打断了一个完整的概念，也不会完全丢失

#### 2.3.3 为什么需要 Embedding Models？

这是 RAG 的"魔法核心"。

**传统关键词搜索的问题：**

```
用户问: "怎么让模型回答更准确？"
文档内容: "提升 LLM 准确性的方法..."

关键词搜索会失败:
  → "让" vs "提升"、"模型" vs "LLM"、"回答" vs "准确性"
  → 字面不匹配，搜不到
```

**嵌入（Embedding）的思路：**

```
把文本转换成一个高维空间中的点（向量）:
  "怎么让模型回答更准确？"     → [0.12, -0.45, 0.78, ...]
  "提升 LLM 准确性的方法"       → [0.11, -0.44, 0.79, ...]
  "今天天气不错"                → [-0.88, 0.23, 0.01, ...]

关键性质:
  → 意思相近的文本在向量空间中距离近
  → 即使用词完全不同
  → 这是通过预训练模型学到的"语义理解"
```

所以 embedding 把"字面匹配"升级成了"语义匹配"——这是 RAG 能够工作的根本原因

#### 2.3.4 为什么需要 Vector Stores？

你有了向量之后，需要能**快速**找出"和某个向量最相似的前 K 个"。

```
朴素做法: 遍历所有向量，算距离，排序
  → 10 万个向量还能接受
  → 1000 万个向量就会慢到无法使用

向量数据库的做法:
  → 用专门的算法（HNSW、IVF 等）建立索引
  → 毫秒级返回近似最邻居
  → 即使亿级向量也能撑住
```

常见的向量数据库：Chroma（本地小规模）、Qdrant、Weaviate、Pinecone（云托管）、pgvector（PostgreSQL 插件）等。LangChain 对它们有统一接口，切换时只改几行代码

#### 2.3.5 为什么需要 Retrievers？

Retriever 是对"检索"这件事的**接口抽象**：

```python
# 所有 retriever 都实现同一个接口
documents = retriever.invoke("用户的问题")
# → 返回一个 Document 列表
```

它可以背后是：
- 向量存储（最常见）
- 关键词搜索引擎（比如 BM25）
- 混合搜索（向量 + 关键词）
- 网络搜索 API
- 甚至是调用另一个 LLM 做检索

**这个抽象的价值：** 你的 Agent 代码只需要和 `retriever` 对话，底层换成什么实现都不影响上层逻辑。这就是第十一节讲过的"上下文工程"在检索领域的体现——**把"信息从哪来"解耦出来**

---

## 第三章：三种 RAG 架构

RAG 不是一种架构，而是一类架构。根据**谁在控制检索的时机**，可以分成三种主流形态

### 3.1 快速对比

| 架构 | 描述 | 控制度 | 灵活性 | 延迟 | 典型场景 |
|------|------|:---:|:---:|:---:|---------|
| **2-Step RAG** | 先检索后生成，流程固定 | ✅ 高 | ❌ 低 | ⚡ 快 | FAQ 机器人、文档问答 |
| **Agentic RAG** | LLM Agent 自主决定何时检索 | ❌ 低 | ✅ 高 | ⏳ 可变 | 有多个数据源的研究助手 |
| **Hybrid RAG** | 混合两者 + 验证步骤 | ⚖️ 中 | ⚖️ 中 | ⏳ 可变 | 需要质量控制的领域问答 |

### 3.2 延迟的一个澄清

> **2-Step RAG 的延迟更"可预测"，而不一定更"低"。**

```
2-Step RAG:
  → LLM 调用次数固定（通常就 1 次）
  → 延迟 = 固定的检索时间 + 固定的 1 次 LLM 调用
  → 可预测、可容量规划

Agentic RAG:
  → LLM 可能调用多次（取决于它决定查几次）
  → 延迟 = 不定次检索 + 不定次 LLM 调用
  → 平均更慢，而且波动大
```

**注意：** 延迟不只取决于 LLM，还取决于检索本身的速度（数据库查询、API 响应、网络延迟）。2-Step RAG 的可预测性是**基于 LLM 推理是主要耗时**的假设

---

## 第四章：2-Step RAG——简单直接的经典模式

### 4.1 基本流程

```
用户问题
   ↓
[步骤 1] 检索相关文档（固定发生）
   ↓
[步骤 2] 生成答案（基于检索到的文档）
   ↓
返回答案
```

**核心特征：** 检索**一定会发生**，且**只发生一次**。这让整个流程高度可预测

### 4.2 什么时候用？

```
✓ 问题类型单一（比如就是"关于我们产品文档的问题"）
✓ 检索几乎一定有用（不存在"不需要查资料就能答"的情况）
✓ 对延迟敏感（需要可预测的响应时间）
✓ 需要简单、易于维护的系统

典型场景:
  - 客服 FAQ 机器人
  - 公司内部文档问答
  - 产品手册查询
  - 规章制度咨询
```

### 4.3 简化代码示意

```python
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# 假设已经构建好了 vector store（包含所有文档的向量）
vector_store = InMemoryVectorStore(OpenAIEmbeddings())
# vector_store.add_documents(...)  # 事先加载文档

model = init_chat_model("claude-sonnet-4-6")


def answer_question(question: str) -> str:
    """2-Step RAG 的核心逻辑：先检索，再生成。"""

    # 步骤 1: 固定执行检索
    # similarity_search 返回与 question 语义最相似的前 4 个文档块
    relevant_docs = vector_store.similarity_search(question, k=4)

    # 把检索到的文档拼成 context 字符串
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # 步骤 2: 构造 prompt，让 LLM 基于 context 回答
    prompt = f"""基于以下资料回答问题。如果资料中没有答案，请如实说明。

资料：
{context}

问题：{question}

答案："""

    # 只调用一次 LLM，没有工具、没有循环
    response = model.invoke(prompt)
    return response.content
```

**注意这里没有 Agent！** 2-Step RAG 本质上是一个**线性的 chain**——不需要 `create_agent`，不需要工具调用循环，就是一次检索加一次 LLM 调用

### 4.4 2-Step RAG 的优势和局限

```
优势:
  + 简单：代码量少，bug 少
  + 快速：固定 1 次 LLM 调用
  + 成本低：token 消耗可预估
  + 易调试：流程线性，出问题容易定位

局限:
  - 不灵活：每个问题都强制先检索
  - 无判断：LLM 不能决定"这个问题我不需要查资料"
  - 单源：通常只能查一个知识库
  - 无迭代：一次检索不够也没法再查
```

---

## 第五章：Agentic RAG——让 LLM 自己决定

### 5.1 基本思路

在 Agentic RAG 中，**检索不再是固定步骤，而是 Agent 工具箱里的一个工具**。Agent 通过推理自己决定：

```
- 这个问题需要查资料吗？  （有些问题根本不需要）
- 查哪个知识库？          （可能有多个）
- 怎么查？               （查询词怎么构造）
- 查一次够吗？            （不够就再查）
- 查什么内容够了？         （什么时候停下来作答）
```

这就和"真人研究员"的工作方式一致了。

### 5.2 流程图

```
用户问题
    ↓
   Agent (LLM 推理)
    ↓
需要外部信息吗？
  ├─ 否 ──────────────→ 直接生成答案
  └─ 是 → 调用搜索工具
            ↓
         获得信息
            ↓
         足够回答了吗？
          ├─ 否 → 回到 Agent（可能再查一次）
          └─ 是 → 生成最终答案
                      ↓
                    返回用户
```

### 5.3 实现起来有多简单？

**震惊点：** 实现 Agentic RAG 其实就是"创建一个带检索工具的普通 Agent"——你在前面几章已经学过的所有东西，全都能直接用。

```python
import requests
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


@tool
def fetch_url(url: str) -> str:
    """从指定 URL 获取文本内容。"""
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return response.text


# 一个清晰的系统提示告诉 Agent 什么时候该用这个工具
system_prompt = """
需要从网页获取信息时使用 fetch_url 工具，并引用相关片段。
"""

# 就这样——一个带检索工具的普通 Agent，它就是 Agentic RAG
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[fetch_url],         # 检索工具
    system_prompt=system_prompt,
)
```

**关键理解：** Agentic RAG 不是一个新概念，而是"给 Agent 配检索工具"的别名。Agent 自带的工具调用循环让它天然支持"需要时才检索、不够时再检索"的模式。

### 5.4 实战案例：基于 llms.txt 的文档助手

LangChain 文档提供了一个经典的 Agentic RAG 实例——一个能动态查阅 LangGraph 官方文档回答问题的助手。

**核心思路：**
```
1. llms.txt 是一个约定——网站在根目录提供这个文件，列出所有文档 URL
2. Agent 启动时先加载这个索引（一次性，不需要 LLM 调用）
3. 当用户提问时，Agent 看着这个索引决定"要查哪几个 URL"
4. Agent 调用 fetch_documentation 工具实际拉取文档内容
5. 基于获取的内容生成答案
```

**完整代码：**

```python
import requests
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from markdownify import markdownify


# 白名单：只允许从这些域名拉文档（安全措施）
ALLOWED_DOMAINS = ["https://langchain-ai.github.io/"]
LLMS_TXT = "https://langchain-ai.github.io/langgraph/llms.txt"


@tool
def fetch_documentation(url: str) -> str:
    """获取并转换指定 URL 的文档内容。"""
    # 安全检查：防止 Agent 被指示去访问任意 URL
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            f"错误：URL 不在允许列表中。"
            f"必须以下列域名之一开头：{', '.join(ALLOWED_DOMAINS)}"
        )
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    # markdownify 把 HTML 转成 markdown，对 LLM 更友好
    return markdownify(response.text)


# 在创建 Agent 之前先拉取 llms.txt
# → 这是"索引"，告诉 Agent 都有哪些文档可查
# → 一次性加载，不需要 LLM 调用
llms_txt_content = requests.get(LLMS_TXT).text


# 系统提示中直接嵌入 llms.txt 内容作为 Agent 的"目录"
system_prompt = f"""
你是一个精通 Python 和 LangGraph 的技术助手。

指令：
1. 遇到你不确定的问题，或涉及 API 用法、行为、配置的问题，
   你必须使用 fetch_documentation 工具查阅相关文档。
2. 引用文档时要清晰总结，包含原文相关上下文。
3. 不要使用允许域名之外的任何 URL。
4. 如果文档拉取失败，告诉用户并尽力用你的专业知识回答。

你可以从以下官方来源获取文档：

{llms_txt_content}

回答用户关于 LangGraph 的问题前，你必须先查阅文档获取最新信息。

回答应当清晰、简洁、技术上准确。
"""

tools = [fetch_documentation]

model = init_chat_model("claude-sonnet-4-0", max_tokens=32_000)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    name="Agentic RAG",
)


# 使用示例
response = agent.invoke({
    "messages": [
        HumanMessage(content=(
            "写一个简短的 LangGraph agent 示例，"
            "使用预置的 create_react_agent，"
            "这个 agent 应该能够查询股票价格信息。"
        ))
    ]
})

print(response["messages"][-1].content)
```

**这个例子的精妙之处：**

```
1. 两层索引设计
   - 第一层：llms.txt（轻量目录）直接嵌入 system prompt
   - 第二层：实际文档内容按需通过工具拉取
   → 类似人类看目录找章节、再翻开那章读

2. 安全边界
   - ALLOWED_DOMAINS 白名单
   - 防止提示注入让 Agent 访问恶意 URL
   - 生产环境 Agentic RAG 必须考虑的问题

3. 优雅降级
   - 工具失败时指示 Agent "尽力用专业知识回答"
   - 不会因为单次拉取失败就完全卡死
```

### 5.5 Agentic RAG 的优势和局限

```
优势:
  + 灵活：Agent 可以决定查或不查、查什么、查几次
  + 多源：可以给 Agent 配多个检索工具
  + 迭代：一次不够可以再查
  + 智能：面对简单问题可以跳过检索直接答

局限:
  - 不可预测：延迟和成本因请求而异
  - 可能过度检索：Agent 可能无谓地多查几次
  - 可能不检索：Agent 可能"觉得自己知道"而编造答案
  - 调试难：出问题要看 Agent 的推理轨迹
```

---

## 第六章：Hybrid RAG——带质量控制的混合架构

### 6.1 什么时候需要 Hybrid？

当 2-Step 太死板、Agentic 又太随意时，Hybrid RAG 提供了一个折中方案——**引入中间验证步骤**来保证质量。

典型痛点：

```
场景 1: 用户提问模糊
  "这个怎么弄？"
  → 2-Step RAG 检索"怎么弄" → 搜到一堆不相关的
  → Agentic RAG 可能瞎猜意图
  → Hybrid: 先做查询增强（Query Rewriting）再检索

场景 2: 检索质量不稳定
  检索结果有时包含完全不相关的文档
  → 2-Step RAG 会把垃圾塞进 prompt
  → Hybrid: 检索后做相关性验证，不合格就重查

场景 3: 答案需要严格校验
  医疗、法律、金融场景下，答案必须有依据
  → Hybrid: 生成答案后做验证，不通过就重新生成
```

### 6.2 Hybrid RAG 的典型组件

```
1. 查询增强（Query Enhancement）
   → 改写模糊的问题
   → 生成多个查询变体
   → 补充上下文信息

2. 检索验证（Retrieval Validation）
   → 评估检索结果是否相关、是否足够
   → 不够就重新改写查询并再次检索

3. 答案验证（Answer Validation）
   → 检查答案的准确性、完整性、来源一致性
   → 不合格就重新生成或修正
```

### 6.3 流程图

```
用户问题
   ↓
查询增强 ←─────────┐
   ↓              │
检索文档          │
   ↓              │
信息足够吗？      │
  ├─ 否 → 改写查询┘
  └─ 是
       ↓
    生成答案
       ↓
    答案质量 OK 吗？
      ├─ 否 → 换个思路？
      │         ├─ 是 → 改写查询
      │         └─ 否 → 返回最佳答案
      └─ 是 → 返回最佳答案
                ↓
             返回用户
```

### 6.4 适用场景

```
✓ 查询模糊或不完整（用户可能说不清楚问题）
✓ 需要质量控制（答案要经得起审查）
✓ 多来源、多步骤推理（单次检索不够）
✓ 自我纠错（系统能识别自己答错了）
```

**实现复杂度：** 这已经不是简单的 Agent 或 Chain 能表达的了——需要用 LangGraph 手写一个带条件分支和循环的工作流。这和第二节讲的"Custom Workflow"多智能体模式是一脉相承的。

---

## 第七章：三种架构的选型决策

### 7.1 决策流程图

```
你的场景是...

问题类型单一、需要低延迟？
  → 2-Step RAG ✅

有多种问题类型、需要灵活查不同来源？
  → Agentic RAG ✅

对答案质量有严格要求、需要验证？
  → Hybrid RAG ✅

不确定？
  → 先用 2-Step RAG 做原型
  → 发现不够用再升级到 Agentic
  → 还不够用再引入 Hybrid 的验证步骤
```

### 7.2 渐进式升级路径

一个非常实用的策略是**逐步演进**，不要一上来就搞最复杂的：

```
阶段 1: 2-Step RAG
  → 最快上线，验证"检索能不能帮上忙"
  → 跑两周，收集失败案例

阶段 2: 加入查询增强
  → 发现用户问题太模糊？加一个查询改写步骤
  → 还是 2-Step 结构，但检索质量变好

阶段 3: 升级到 Agentic RAG
  → 发现有多种问题类型？让 Agent 自己判断
  → 添加多个检索工具

阶段 4: 加入验证
  → 发现质量还是不稳定？引入检索验证和答案验证
  → 演化成 Hybrid RAG
```

### 7.3 一个容易忽视的问题：评估

无论你选哪种架构，**都需要评估系统效果**：

```
关键指标:
  - 检索命中率：检索到的文档里有多少真的相关？
  - 答案正确率：生成的答案和标准答案的吻合度
  - 引用准确率：引用的文档是否真的支持答案
  - 延迟和成本：P50、P90、P99 的延迟；单次查询成本
```

没有评估就没有改进——很多 RAG 系统失败不是架构错了，而是没人能说出"到底哪里错了"。

---

## 第八章：概念串联

### 8.1 知识地图

```
第一节（Agent）
  → Agentic RAG 就是"带检索工具的 Agent"
  → 你已经学会的所有 Agent 知识都直接适用

第四节（Tools）
  → 检索函数包装成 @tool 就成了 Agentic RAG 的核心
  → fetch_documentation、search_knowledge_base 都是普通工具

第十一节（Context Engineering）
  → RAG 本质上是一种上下文注入策略
  → "检索到的文档"是注入到 LLM 上下文的新信息
  → 2-Step RAG 是静态注入，Agentic RAG 是动态注入

第十二节（MCP）
  → MCP 服务器可以提供检索能力
  → 比如一个"文档检索 MCP 服务器"让任何 Agent 即插即用

第十三节（Multi-agent）
  → Subagents 模式下，研究子 Agent 往往就是一个 Agentic RAG
  → 主 Agent 把"研究某主题"的任务交给它

本节（Retrieval & RAG）← 你在这里
  → 给 Agent 装上"查资料"的能力
  → 三种架构对应不同的控制度/灵活性权衡
```

### 8.2 工具视角 vs Chain 视角

RAG 有两种实现心智：

```
Chain 视角（2-Step RAG）:
  数据流: 输入 → 检索 → 生成 → 输出
  控制: 代码层面硬编码
  适合: 流程固定的场景

Tool 视角（Agentic RAG）:
  数据流: Agent 推理 → [调检索工具] → Agent 推理 → ...
  控制: LLM 层面自主决策
  适合: 流程灵活的场景

理解这两个视角的区别，就理解了 RAG 的全部设计空间。
```

---

## 第九章：最佳实践与常见陷阱

### 9.1 五条最佳实践

```
1. 优先用已有的知识库
   → 不要为了用 RAG 而从零建一个向量库
   → 你公司的 SQL 数据库、搜索引擎、CRM 都是现成的"知识库"
   → 包装成工具就能用

2. 切块策略要实验
   → 没有"标准答案"——500、1000、2000 字符各有适用场景
   → 必须用真实查询评估不同策略的效果

3. metadata 是金矿
   → 在 Document 的 metadata 里存来源、时间、作者等
   → 检索后可以用于过滤、排序、引用
   → 也可以让 LLM 告诉用户"这个答案来自 xxx 文档第 x 页"

4. Agentic RAG 要给足指引
   → 系统提示里明确说"什么时候该查、什么时候不用查"
   → 否则 Agent 可能过度检索或完全不检索

5. 生产环境加白名单
   → Agentic RAG 的 URL/查询参数可能被提示注入攻击
   → 必须限制工具能访问的资源范围
```

### 9.2 五个常见陷阱

```
陷阱 1: 以为"嵌入模型越大越好"
  → 更大的嵌入模型并不一定检索更准
  → 要在你的实际数据和查询上测试

陷阱 2: 忽视评估数据集的重要性
  → "感觉不错"不是评估
  → 至少准备 50-100 个带标准答案的问题，定期跑一遍

陷阱 3: chunk 切得太大或太小
  → 太小：每块没有足够上下文
  → 太大：检索不精确，浪费 token
  → 实验是唯一的答案

陷阱 4: 把所有东西都塞进 prompt
  → 检索到 20 个文档全都塞进去？LLM 会"迷路"
  → 做 rerank，只留最相关的 3-5 个

陷阱 5: 忘了知识更新
  → 知识库不是一次性构建的
  → 文档变了、新增了，向量库也要同步更新
  → 否则 RAG 回答的是"昨天的知识"
```

---

## 第十章：速查手册

### 10.1 RAG 架构选型

| 需求 | 推荐架构 |
|------|---------|
| 最快上线、场景单一 | 2-Step RAG |
| 多数据源、需要灵活判断 | Agentic RAG |
| 对答案质量有严格要求 | Hybrid RAG |
| 不确定 | 从 2-Step 开始，按需升级 |

### 10.2 核心组件速查

| 组件 | 作用 | 关键选择 |
|------|------|---------|
| Document Loader | 加载数据源 | 按数据源类型选（PDF/Web/Notion…） |
| Text Splitter | 切块 | chunk_size、overlap |
| Embedding Model | 文本向量化 | 质量 vs 成本 vs 速度 |
| Vector Store | 存储和检索向量 | 规模决定（本地 vs 云托管） |
| Retriever | 统一检索接口 | 向量 / 关键词 / 混合 |

### 10.3 最小 2-Step RAG 模板

```python
# 1. 检索
docs = vector_store.similarity_search(question, k=4)
context = "\n\n".join(d.page_content for d in docs)

# 2. 生成
prompt = f"基于资料回答:\n{context}\n\n问题: {question}"
answer = model.invoke(prompt).content
```

### 10.4 最小 Agentic RAG 模板

```python
@tool
def search_knowledge(query: str) -> str:
    """检索知识库。"""
    docs = vector_store.similarity_search(query, k=4)
    return "\n\n".join(d.page_content for d in docs)


agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[search_knowledge],
    system_prompt="需要查资料时调用 search_knowledge 工具。",
)
```

### 10.5 关键 API

| API | 作用 |
|-----|------|
| `Document(page_content, metadata)` | 标准文档对象 |
| `text_splitter.split_documents(docs)` | 切块 |
| `embeddings.embed_documents(texts)` | 批量向量化 |
| `vector_store.add_documents(docs)` | 入库 |
| `vector_store.similarity_search(query, k)` | 相似度检索 |
| `retriever.invoke(query)` | 统一检索接口 |

---

## 结语

RAG 是 LLM 应用走向实用的关键一步——它把"全能但不懂你业务"的通用模型变成了"懂你业务"的专属助手。但**不要神化 RAG**：它本质上就是"先查资料再回答"这个朴素思想的工程实现。

**核心心智模型：**

```
LLM 的记忆 = 百科全书（广博但固定）
RAG 的检索 = 图书馆（专精且可更新）

  → 简单问题用百科全书就够（直接问 LLM）
  → 专业问题要查图书馆（RAG）
  → 真正强大的系统是两者结合：
     有常识的 LLM + 有专业资料的检索 = 可靠的回答
```

三种架构的选择归根结底是在**控制度、灵活性、延迟**三者之间做权衡——没有"最好的"架构，只有"最适合你当前场景的"架构。从简单开始，按需演进，用评估驱动改进，这就是构建 RAG 系统的正道。
