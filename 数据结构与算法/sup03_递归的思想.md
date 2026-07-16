# 彻底搞懂递归 —— 从思维模型到实战解题全攻略

---

## 写在前面：你为什么觉得递归难？

大多数人学递归时会犯一个致命错误：**试图用大脑模拟计算机，一层一层地"跟踪"递归调用的展开过程**。比如看到 `f(5) → f(4) → f(3) → f(2) → f(1) → 回溯……`，你会试着在脑子里把每一层都展开，然后跟踪返回值一层层传回来。当层数超过 3-4 层时，你的"脑内栈"就溢出了

> **核心观念转变**：递归的正确思维方式不是"展开"，而是**"信任"**
>
> 你不需要知道子问题内部是怎么解决的，你只需要——
> 1. **相信**你调用的那个函数能正确完成它的任务（因为它跟你写的是同一个函数）
> 2. 聚焦于：**当前这一层，我该做什么？**

---

## 第一章：递归的本质 —— 不是"自己调自己"

### 1.1 递归的真正定义

很多教材说"递归就是函数调用自身"。这只是**语法层面**的描述，不是本质

递归的本质是一种**问题分解策略**：

$$
\text{大问题} = \text{当前层的一小步操作} + \text{规模更小的同类子问题}
$$
换句话说，递归是在回答这样一个问题：

> **"如果规模更小的子问题已经被某个人解决了，我只需要再做一点点什么，就能解决当前规模的问题？"**

### 1.2 一个生活类比

假设你站在一列队伍的最后面，想知道自己排在第几个。你不需要从头数到尾，而是可以：

1. 拍一下前面那个人的肩膀，问："你排第几个？"
2. 前面那个人不知道，他也拍前面的人的肩膀问同样的问题
3. 一直传递到第一个人，第一个人说："我是第 1 个。"（**基线条件**）
4. 第二个人听到后说："那我是第 2 个。"然后告诉后面的人。（**回溯**）
5. 一层层传回来，最终你知道了自己排第几

```python
def position(person):
    """问当前这个人：你排第几？"""
    if person.前面没有人():  # 基线条件
        return 1
    return position(person.前面那个人) + 1  # 递归 + 当前层操作
```

你没有"展开"这条队伍的每一个人然后从头数到尾——你只是**信任**前面那个人会给你正确答案，然后 +1

### 1.3 递归思维的三个核心支柱

无论多复杂的递归题，都可以拆解为以下三步：

```css
┌──────────────────────────────────────────────────────────────┐
│                    递归三要素（黄金框架）                       │
│                                                              │
│  ① 基线条件（Base Case）                                      │
│     → 什么时候可以直接给出答案，不再递归？                       │
│     → 这是递归的"出口"，缺少它会导致无限递归。                   │
│                                                              │
│  ② 递归关系（Recursive Relation）                             │
│     → 当前问题如何分解为规模更小的同类子问题？                    │
│     → 子问题的结果如何与当前层操作组合得到最终结果？               │
│                                                              │
│  ③ 返回值（Return Value）                                     │
│     → 每一层递归应该向上一层返回什么信息？                       │
│     → 这决定了函数签名（参数和返回类型）。                       │
└──────────────────────────────────────────────────────────────┘
```

> **做递归题之前，先在纸上或注释里明确写出这三条**

---

## 第二章：从最简单的例子开始，建立"信任"

### 2.1 例题：计算 n 的阶乘

**问题**：计算$ n! = n \times (n-1) \times \cdots \times 1$

**第一步：用标准框架分析**

```css
① 基线条件：n == 0 或 n == 1 时，阶乘是 1。
② 递归关系：n! = n × (n-1)!
             "如果有人告诉了我 (n-1)!，我只需要乘以 n 就行了。"
③ 返回值：  当前 n 对应的阶乘值（一个整数）。
```

**第二步：直接翻译为代码**

```python
def factorial(n: int) -> int:
    """计算 n 的阶乘。"""
    # ① 基线条件
    if n <= 1:
        return 1
    # ② 递归关系：信任 factorial(n-1) 能给出正确结果
    return n * factorial(n - 1)
```

**第三步：验证（不要在脑子里展开！用"信任法"验证）**

- `factorial(1)` → 命中基线，返回 1 ✓
- `factorial(2)` → "信任 `factorial(1)` 返回 1"，当前层做 `2 × 1 = 2` ✓
- `factorial(3)` → "信任 `factorial(2)` 返回 2"，当前层做 `3 × 2 = 6` ✓

我们只验证了"当前层的逻辑是否正确"，而不是把整棵递归树都展开

### 2.2 例题：反转字符串

**问题**：将字符串 `"hello"` 反转为 `"olleh"`。

**黄金框架分析**

```
① 基线条件：字符串长度 ≤ 1，直接返回自身。
② 递归关系：reverse("hello") = reverse("ello") + "h"
             "如果有人告诉了我 ello 的反转是 olle，我只需要把 h 接在后面。"
③ 返回值：  反转后的字符串。
```

```python
def reverse_string(s: str) -> str:
    """递归反转字符串。"""
    if len(s) <= 1:
        return s
    # 信任 reverse_string(s[1:]) 能正确反转剩余部分
    return reverse_string(s[1:]) + s[0]

print(reverse_string("hello"))  # "olleh"
```

### 2.3 例题：求链表长度

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def length(head: ListNode | None) -> int:
    """递归计算链表长度。"""
    # ① 基线条件：空链表长度为 0
    if head is None:
        return 0
    # ② 递归关系："信任 length(head.next) 能告诉我后面有几个节点"
    #    当前层操作：加上自己这一个
    return 1 + length(head.next)
```

---

## 第三章：递归的执行机制 —— 调用栈

虽然我们强调"不要在脑中展开递归"，但**理解调用栈的工作原理**对于调试和分析复杂度至关重要

### 3.1 函数调用栈

每次函数被调用时，计算机会在**调用栈（Call Stack）**上创建一个新的**栈帧（Stack Frame）**，存储：

- 函数的参数
- 局部变量
- 返回地址（调用结束后回到哪里继续执行）

递归调用就是不断地往栈上压入新的栈帧，直到触发基线条件后，再一帧一帧地弹出（回溯）

### 3.2 图解 `factorial(4)` 的调用栈

```css
调用阶段（递 / "递推"）                    回溯阶段（归 / "回归"）
========================                 ========================

    ┌──────────────────┐
    │ factorial(1)     │ → 命中基线，返回 1
    │   return 1       │ ─────────────────→  1
    ├──────────────────┤                     ↓
    │ factorial(2)     │                   2 × 1 = 2
    │   return 2 * ?   │ ←────────────────   2
    ├──────────────────┤                     ↓
    │ factorial(3)     │                   3 × 2 = 6
    │   return 3 * ?   │ ←────────────────   6
    ├──────────────────┤                     ↓
    │ factorial(4)     │                   4 × 6 = 24
    │   return 4 * ?   │ ←────────────────   24
    └──────────────────┘

栈的最大深度 = 4（等于 n）
```

### 3.3 递归的两个阶段

每个递归函数的执行天然分为两个阶段：

```python
def f(n):
    # ===== 递推阶段 =====
    # （递归调用之前的代码）
    # 这里的操作在"向下走"的过程中执行

    result = f(n - 1)  # ← 递归调用

    # ===== 回归阶段 =====
    # （递归调用之后的代码）
    # 这里的操作在"向上回来"的过程中执行

    return result
```

**这个区分非常重要**，因为许多题目需要你在不同阶段执行不同操作：

```python
def print_forward_then_backward(n: int) -> None:
    """先正序打印 1→n，再倒序打印 n→1。"""
    if n == 0:
        return
    print(f"递推阶段: {n}")      # 递推时执行（先打印大的）
    print_forward_then_backward(n - 1)
    print(f"回归阶段: {n}")      # 回归时执行（先打印小的）

print_forward_then_backward(3)
# 输出:
# 递推阶段: 3
# 递推阶段: 2
# 递推阶段: 1
# 回归阶段: 1
# 回归阶段: 2
# 回归阶段: 3
```

> 这就是为什么递归可以"自然地"实现逆序操作——**回归阶段天然是逆序的**

### 3.4 Python 递归深度限制

Python 默认的最大递归深度为 1000。当递归深度可能超过这个限制时：

```python
import sys
sys.setrecursionlimit(10000)  # 可以调大，但不推荐过大
```

更好的方案是在需要时将递归改写为迭代（后面第七章会专门讲）

---

## 第四章：递归解题的系统方法论

### 4.1 解题五步法（核心方法，建议背下来）

当你面对一道需要递归的题时，按以下顺序思考：

```css
步骤 1 ▸ 明确函数的定义
         这个函数接收什么参数？返回什么？
         用一句话描述：这个函数 DO WHAT？

步骤 2 ▸ 确定基线条件
         输入规模最小 / 最简单的情况是什么？答案是什么？
         通常是：空节点、空列表、n=0、n=1、只有一个元素……

步骤 3 ▸ 假设子问题已解决（"信任"子调用）
         假设对于规模更小的输入，这个函数已经返回了正确结果。
         不要去想它内部怎么做到的。

步骤 4 ▸ 用子问题的结果构建当前问题的答案
         这一步是最核心的逻辑。
         通常就是 1-3 行代码。

步骤 5 ▸ 检验
         用最小的非基线用例（比如 n=2 或只有 2-3 个节点的树）
         验证逻辑是否正确。
```

### 4.2 实战演示：二叉树的最大深度

**题目**：给定一棵二叉树，返回其最大深度（根到最远叶子节点的路径上的节点数）。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

**按五步法思考**：

```
步骤 1 ▸ 函数定义
         maxDepth(root) 接收一棵树的根节点，返回这棵树的最大深度。

步骤 2 ▸ 基线条件
         root 为 None（空树）→ 深度为 0。

步骤 3 ▸ 信任子调用
         假设 maxDepth(root.left) 已经正确返回了左子树的最大深度。
         假设 maxDepth(root.right) 已经正确返回了右子树的最大深度。

步骤 4 ▸ 构建当前答案
         整棵树的最大深度 = max(左子树深度, 右子树深度) + 1
                                                        ↑ 加上根节点自己这一层

步骤 5 ▸ 验证
         只有根节点的树: max(0, 0) + 1 = 1 ✓
         根+左子节点: max(1, 0) + 1 = 2 ✓
```

```python
def max_depth(root: TreeNode | None) -> int:
    """LeetCode 104: 二叉树的最大深度。"""
    if root is None:
        return 0
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    return max(left_depth, right_depth) + 1
```

就这么简单。**四行代码**。没有任何"展开"递归的过程

### 4.3 实战演示：翻转二叉树

**题目**：翻转一棵二叉树（每个节点的左右子树互换）。

```css
     4                 4
   /   \      →      /   \
  2     7           7     2
 / \   / \         / \   / \
1   3 6   9       9   6 3   1
```

**五步法**：

```css
步骤 1 ▸ invertTree(root) → 翻转以 root 为根的整棵树，返回翻转后的根节点。

步骤 2 ▸ root 为 None → 返回 None。

步骤 3 ▸ 信任 invertTree(root.left) 已经把左子树翻转好了。
         信任 invertTree(root.right) 已经把右子树翻转好了。

步骤 4 ▸ 左右子树各自已翻转完毕，现在只需要交换它们的位置。
         root.left, root.right = root.right, root.left

步骤 5 ▸ 单节点树: 无子树可交换 → 返回自身 ✓
         三节点树: 左右子树递归返回自身（叶子），然后交换 → 正确 ✓
```

```python
def invert_tree(root: TreeNode | None) -> TreeNode | None:
    """LeetCode 226: 翻转二叉树。"""
    if root is None:
        return None
    # 信任：这两个调用各自完成了翻转
    left_inverted = invert_tree(root.left)
    right_inverted = invert_tree(root.right)
    # 当前层操作：交换
    root.left = right_inverted
    root.right = left_inverted
    return root
```

---

## 第五章：递归在二叉树中的四大范式

刷 LeetCode 二叉树题时，几乎所有递归都可以归入以下四种模式之一

### 5.1 范式一：自顶向下（Top-Down / 前序位置）

**特点**：先处理当前节点，再递归处理子树。信息从上往下传递（通过参数传递）

```python
def top_down(node, 从上面传下来的信息):
    if node is None:
        return
    # 先处理当前节点
    做一些事情(node, 从上面传下来的信息)
    # 再递归
    top_down(node.left, 更新后的信息)
    top_down(node.right, 更新后的信息)
```

**典型题目**：二叉树的所有路径（LeetCode 257）

```python
def binary_tree_paths(root: TreeNode | None) -> list[str]:
    """LeetCode 257: 返回所有从根到叶子的路径。"""
    result: list[str] = []

    def dfs(node: TreeNode | None, path: str) -> None:
        if node is None:
            return
        # 当前层：把自己加到路径中
        current_path = f"{path}->{node.val}" if path else str(node.val)

        # 如果是叶子节点，收集结果
        if node.left is None and node.right is None:
            result.append(current_path)
            return

        # 递归：把更新后的路径传下去
        dfs(node.left, current_path)
        dfs(node.right, current_path)

    dfs(root, "")
    return result
```

### 5.2 范式二：自底向上（Bottom-Up / 后序位置）

**特点**：先递归处理子树，拿到子问题的结果，再处理当前节点。信息从下往上传递（通过返回值传递）。

```python
def bottom_up(node) -> 某种结果:
    if node is None:
        return 基线值
    # 先递归
    left_result = bottom_up(node.left)
    right_result = bottom_up(node.right)
    # 再处理当前节点（利用子问题的结果）
    return 合并(left_result, right_result, node)
```

**典型题目**：判断是否平衡二叉树（LeetCode 110）

```python
def is_balanced(root: TreeNode | None) -> bool:
    """LeetCode 110: 判断是否为平衡二叉树。"""

    def height(node: TreeNode | None) -> int:
        """返回树的高度；若不平衡则返回 -1。"""
        if node is None:
            return 0

        # 信任子调用
        left_h = height(node.left)
        if left_h == -1:
            return -1  # 左子树已不平衡，提前剪枝

        right_h = height(node.right)
        if right_h == -1:
            return -1  # 右子树已不平衡

        # 当前层判断
        if abs(left_h - right_h) > 1:
            return -1  # 当前节点不平衡

        return max(left_h, right_h) + 1

    return height(root) != -1
```

> **技巧**：用特殊返回值（如 -1）在回溯过程中"提前传递失败信息"，避免重复计算。

### 5.3 范式三：分治法（Divide and Conquer）

**特点**：将问题拆为左右两个独立子问题，各自解决后合并结果。它是自底向上的特殊形式。

```python
def divide_and_conquer(node):
    if node is None:
        return 基线值
    # 分（Divide）
    left_result = divide_and_conquer(node.left)
    right_result = divide_and_conquer(node.right)
    # 合（Conquer / Merge）
    return merge(left_result, right_result, node)
```

**典型题目**：二叉树的直径（LeetCode 543）

```
直径 = 某个节点的 左子树深度 + 右子树深度 的最大值
```

```python
def diameter_of_binary_tree(root: TreeNode | None) -> int:
    """LeetCode 543: 二叉树的直径。"""
    max_diameter = 0

    def depth(node: TreeNode | None) -> int:
        nonlocal max_diameter
        if node is None:
            return 0
        # 分：求左右子树深度
        left_d = depth(node.left)
        right_d = depth(node.right)
        # 合：经过当前节点的直径 = 左深度 + 右深度
        max_diameter = max(max_diameter, left_d + right_d)
        # 返回：当前子树的深度
        return max(left_d, right_d) + 1

    depth(root)
    return max_diameter
```

> **注意**：这道题的巧妙之处在于函数的"返回值"（深度）和"我们要求的答案"（直径）是**不同的东西**。我们用 `nonlocal` 变量在回溯过程中"顺便"更新答案。

### 5.4 范式四：回溯法（Backtracking）

**特点**：在递归的基础上增加"撤销选择"操作，用于探索所有可能的路径/组合。

```python
def backtrack(node, 当前状态):
    if 满足结束条件:
        收集结果(当前状态)
        return
    for 选择 in 所有可能的选择:
        做出选择(修改当前状态)
        backtrack(下一层, 当前状态)
        撤销选择(恢复当前状态)    # ← 这一步是回溯的核心
```

**典型题目**：路径总和 II（LeetCode 113）

```python
def path_sum(root: TreeNode | None, target_sum: int) -> list[list[int]]:
    """LeetCode 113: 找出所有从根到叶子的路径，使其节点值之和等于目标值。"""
    result: list[list[int]] = []

    def backtrack(node: TreeNode | None, remain: int, path: list[int]) -> None:
        if node is None:
            return

        path.append(node.val)        # 做出选择

        # 叶子节点且剩余值恰好为 0
        if node.left is None and node.right is None and remain == node.val:
            result.append(path[:])   # path[:] 是拷贝，避免后续修改影响结果

        backtrack(node.left, remain - node.val, path)
        backtrack(node.right, remain - node.val, path)

        path.pop()                    # 撤销选择（回溯！）

    backtrack(root, target_sum, [])
    return result
```

### 5.5 四大范式速查表

| 范式 | 信息流向 | 处理时机 | 典型场景 | 代表题目 |
|------|---------|---------|---------|---------|
| 自顶向下 | 参数传递 ↓ | 先处理再递归 | 路径记录、深度传递 | 257, 112 |
| 自底向上 | 返回值传递 ↑ | 先递归再处理 | 高度、节点数、判断 | 104, 110, 98 |
| 分治法 | 返回值合并 ↑ | 左右分别递归后合并 | 构建树、求直径 | 543, 105, 106 |
| 回溯法 | 状态+撤销 ↕ | 探索所有分支 | 组合/排列/路径枚举 | 113, 46, 78 |

---

## 第六章：递归中的常见陷阱与调试技巧

### 6.1 陷阱一：基线条件遗漏或不充分

```python
# ❌ 错误：忘记处理 None
def max_depth(root: TreeNode) -> int:
    # 如果 root 是 None，直接崩溃
    return max(max_depth(root.left), max_depth(root.right)) + 1

# ✅ 正确
def max_depth(root: TreeNode | None) -> int:
    if root is None:  # 基线条件
        return 0
    return max(max_depth(root.left), max_depth(root.right)) + 1
```

> **原则**：基线条件必须覆盖"最小输入"的所有可能情况。

### 6.2 陷阱二：忘记 return

```python
# ❌ 错误：忘记 return，导致返回 None
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    is_same_tree(p.left, q.left)     # 忘记 return！
    is_same_tree(p.right, q.right)   # 忘记 return！

# ✅ 正确
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
```

### 6.3 陷阱三：修改了共享的可变对象

```python
# ❌ 错误：所有路径共享同一个 path 列表
def all_paths(root, path, result):
    if root is None:
        return
    path.append(root.val)
    if not root.left and not root.right:
        result.append(path)       # 这里存的是引用！后续 pop 会影响它
    all_paths(root.left, path, result)
    all_paths(root.right, path, result)
    path.pop()

# ✅ 正确：收集结果时拷贝
def all_paths(root, path, result):
    if root is None:
        return
    path.append(root.val)
    if not root.left and not root.right:
        result.append(path[:])    # path[:] 创建副本
    all_paths(root.left, path, result)
    all_paths(root.right, path, result)
    path.pop()
```

### 6.4 陷阱四：递归方向不对，问题规模没有缩小

```python
# ❌ 错误：n 没有变小，无限递归
def bad_factorial(n):
    if n == 1:
        return 1
    return n * bad_factorial(n)  # 应该是 n-1 !!!

# ❌ 错误：忘记移动到下一个节点
def bad_length(head):
    if head is None:
        return 0
    return 1 + bad_length(head)  # 应该是 head.next !!!
```

### 6.5 调试递归的实用技巧

#### 技巧 1：打印缩进日志

```python
def max_depth_debug(root: TreeNode | None, level: int = 0) -> int:
    """带调试日志的最大深度函数。"""
    indent = "  " * level
    if root is None:
        print(f"{indent}max_depth(None) → 0")
        return 0

    print(f"{indent}max_depth({root.val}) 进入")
    left = max_depth_debug(root.left, level + 1)
    right = max_depth_debug(root.right, level + 1)
    result = max(left, right) + 1
    print(f"{indent}max_depth({root.val}) → {result}")
    return result
```

对于树 `[3, 9, 20]`，输出：

```
max_depth(3) 进入
  max_depth(9) 进入
    max_depth(None) → 0
    max_depth(None) → 0
  max_depth(9) → 1
  max_depth(20) 进入
    max_depth(None) → 0
    max_depth(None) → 0
  max_depth(20) → 1
max_depth(3) → 2
```

#### 技巧 2：用最小用例手动走一遍

不要用大例子调试。只用 0-3 个节点的树。如果最小用例都对了，大概率逻辑是正确的。

#### 技巧 3：检查函数的"合同"

把你的递归函数想象成一份"合同"：

- **前置条件**：调用者必须传入什么样的参数？
- **后置条件**：函数承诺返回什么？
- **不变量**：每次递归调用，问题规模是否严格缩小？

如果"合同"在每一层都被遵守，递归就一定是正确的。

---

## 第七章：递归与迭代的转换

### 7.1 为什么要转换？

- **栈溢出风险**：Python 默认递归深度 1000，深度过大会 `RecursionError`。
- **性能考虑**：函数调用有开销（创建栈帧、保存/恢复寄存器）。
- **面试要求**：有时面试官会要求你把递归改成迭代。

### 7.2 尾递归（Tail Recursion）

如果递归调用是函数的**最后一个操作**（即递归调用的结果直接作为函数返回值，不做任何额外运算），就叫做**尾递归**。

```python
# 非尾递归：递归返回后还要做乘法运算
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # 返回后还要 × n

# 尾递归：递归调用是最后一步操作，结果直接返回
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, accumulator * n)  # 直接返回递归结果
```

> **注意**：Python **不优化尾递归**（不会自动转为循环），所以在 Python 中尾递归的主要价值是**方便你手动转为迭代**。

### 7.3 尾递归 → 迭代的机械转换法

尾递归可以直接用 `while` 循环替换：

```python
# 尾递归版本
def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, acc * n)

# 机械转换为迭代
def factorial_iter(n):
    acc = 1                   # 对应尾递归的 accumulator 参数
    while n > 1:              # 对应基线条件的反面
        acc = acc * n         # 对应递归调用前的参数更新
        n = n - 1
    return acc
```

**转换规则**：
1. 函数参数 → 循环变量
2. 基线条件 → 循环终止条件（取反）
3. 递归调用的参数更新 → 循环体内的变量更新

### 7.4 通用递归 → 迭代：手动模拟栈

对于**非尾递归**（比如二叉树的遍历），需要手动用一个栈来模拟调用栈。

#### 前序遍历：递归 vs 迭代

```python
# 递归版
def preorder_recursive(root: TreeNode | None) -> list[int]:
    if root is None:
        return []
    return [root.val] + preorder_recursive(root.left) + preorder_recursive(root.right)

# 迭代版（手动用栈模拟）
def preorder_iterative(root: TreeNode | None) -> list[int]:
    if root is None:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        # 先压右，再压左（因为栈是后进先出，要先处理左）
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```

#### 中序遍历：迭代版

中序遍历的迭代版稍微复杂，因为要"先一路走到最左，再回头处理"：

```python
def inorder_iterative(root: TreeNode | None) -> list[int]:
    result = []
    stack = []
    current = root
    while current or stack:
        # 一路向左走到底
        while current:
            stack.append(current)
            current = current.left
        # 弹出（回溯），处理当前节点
        current = stack.pop()
        result.append(current.val)
        # 转向右子树
        current = current.right
    return result
```

**理解方法**：这段代码模拟的就是递归中的——
- `while current` 循环 = 递归不断调用 `inorder(node.left)` 的过程
- `stack.pop()` = 某一层递归返回的时刻
- `current = current.right` = 处理完左和根后，转向右子树

---

## 第八章：递归复杂度分析

### 8.1 时间复杂度：递推公式法

对于递归函数，时间复杂度通常通过**递推关系（Recurrence Relation）**来分析。

#### 例 1：二叉树遍历

每个节点恰好访问一次：

\[
T(n) = T(\text{left}) + T(\text{right}) + O(1) = O(n)
\]

#### 例 2：归并排序

每次将问题分为两半，合并耗时 \(O(n)\)：

\[
T(n) = 2T\left(\frac{n}{2}\right) + O(n)
\]

根据**主定理（Master Theorem）**，\(T(n) = O(n \log n)\)。

#### 例 3：斐波那契数列（朴素递归）

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```

\[
T(n) = T(n-1) + T(n-2) + O(1) \approx O(2^n) \quad \text{（指数级，非常慢！）}
\]

### 8.2 空间复杂度：递归深度

递归的空间复杂度 = **调用栈的最大深度** × **每帧的空间**。

| 场景 | 最大递归深度 | 空间复杂度 |
|------|------------|-----------|
| 平衡二叉树遍历 | \(O(\log n)\) | \(O(\log n)\) |
| 链表 / 倾斜二叉树遍历 | \(O(n)\) | \(O(n)\) |
| 归并排序 | \(O(\log n)\) | \(O(\log n)\)（不含额外数组） |
| 朴素斐波那契 | \(O(n)\) | \(O(n)\) |

### 8.3 用递归树可视化复杂度

以 `fib(5)` 为例，画出递归树：

```
                     fib(5)
                   /        \
              fib(4)          fib(3)
             /     \          /    \
          fib(3)   fib(2)  fib(2)  fib(1)
          /   \    /   \    /   \
       fib(2) fib(1) fib(1) fib(0) fib(1) fib(0)
       / \
   fib(1) fib(0)
```

可以观察到：
- 大量重复计算（`fib(3)` 算了 2 次，`fib(2)` 算了 3 次）。
- 这就是为什么朴素递归是 \(O(2^n)\)。
- **解决方案**：记忆化（见第九章）。

---

## 第九章：递归优化 —— 记忆化与动态规划

### 9.1 问题：重叠子问题

当递归过程中**同一个子问题被反复求解**时，就出现了"重叠子问题"。这是递归效率低下的主要原因。

### 9.2 解决方案一：记忆化递归（自顶向下 DP）

思路：用一个字典/数组缓存已经计算过的结果。递归前先查缓存。

```python
from functools import lru_cache

# 方法 1：使用 functools.lru_cache 装饰器（最简洁）
@lru_cache(maxsize=None)
def fib(n: int) -> int:
    """斐波那契数列（记忆化递归版）。"""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# 方法 2：手动记忆化（面试中更常用，因为更直观）
def fib_memo(n: int, memo: dict[int, int] | None = None) -> int:
    """斐波那契数列（手动记忆化版）。"""
    if memo is None:
        memo = {}
    if n <= 1:
        return n
    if n in memo:
        return memo[n]      # 命中缓存，直接返回
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

**优化效果**：

| 版本 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 朴素递归 | \(O(2^n)\) | \(O(n)\) |
| 记忆化递归 | \(O(n)\) | \(O(n)\) |

### 9.3 解决方案二：自底向上迭代（经典 DP）

```python
def fib_dp(n: int) -> int:
    """斐波那契数列（自底向上动态规划版）。"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# 空间优化版：只需要前两个值
def fib_optimized(n: int) -> int:
    """斐波那契数列（空间优化版）。"""
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1
```

### 9.4 记忆化在树题中的应用：打家劫舍 III（LeetCode 337）

**题目**：在一棵二叉树中选择不相邻的节点（父子不能同时选），使得选中节点的值之和最大。

```python
def rob(root: TreeNode | None) -> int:
    """LeetCode 337: 打家劫舍 III。"""

    def dfs(node: TreeNode | None) -> tuple[int, int]:
        """返回 (选当前节点的最大值, 不选当前节点的最大值)。"""
        if node is None:
            return (0, 0)

        left_rob, left_skip = dfs(node.left)
        right_rob, right_skip = dfs(node.right)

        # 选当前节点：左右子节点都不能选
        rob_current = node.val + left_skip + right_skip
        # 不选当前节点：左右子节点可选可不选，取最大
        skip_current = max(left_rob, left_skip) + max(right_rob, right_skip)

        return (rob_current, skip_current)

    return max(dfs(root))
```

> **技巧**：通过让函数返回**一个元组**（多个信息），避免重复遍历。这比用 `memo` 字典更优雅。

---

## 第十章：LeetCode 经典递归题精讲

### 10.1 相同的树（LeetCode 100）

```
步骤 1 ▸ is_same(p, q) → 判断两棵树是否完全相同
步骤 2 ▸ p 和 q 都为 None → True；一个为 None 一个不为 → False
步骤 3 ▸ 信任 is_same(p.left, q.left) 和 is_same(p.right, q.right)
步骤 4 ▸ 当前层：p.val == q.val 且左右子树相同 → True
```

```python
def is_same_tree(p: TreeNode | None, q: TreeNode | None) -> bool:
    """LeetCode 100: 相同的树。"""
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (
        p.val == q.val
        and is_same_tree(p.left, q.left)
        and is_same_tree(p.right, q.right)
    )
```

### 10.2 对称二叉树（LeetCode 101）

**思维转换**：不要想"一棵树是否对称"，而是想"两棵子树是否互为镜像"。

```python
def is_symmetric(root: TreeNode | None) -> bool:
    """LeetCode 101: 对称二叉树。"""

    def is_mirror(t1: TreeNode | None, t2: TreeNode | None) -> bool:
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (
            t1.val == t2.val
            and is_mirror(t1.left, t2.right)   # 左的左 vs 右的右
            and is_mirror(t1.right, t2.left)    # 左的右 vs 右的左
        )

    return is_mirror(root, root)
```

### 10.3 从前序与中序遍历构造二叉树（LeetCode 105）

这是**分治法**的经典应用。

```
前序: [3, 9, 20, 15, 7]    →  根 = 3
中序: [9, 3, 15, 20, 7]    →  3 左边是左子树 [9]，右边是右子树 [15,20,7]
```

```python
def build_tree(preorder: list[int], inorder: list[int]) -> TreeNode | None:
    """LeetCode 105: 从前序与中序遍历序列构造二叉树。"""
    if not preorder:
        return None

    # 前序的第一个元素一定是根
    root_val = preorder[0]
    root = TreeNode(root_val)

    # 在中序中找到根的位置，将中序分为左右两部分
    mid = inorder.index(root_val)

    # 分治递归
    # 左子树的前序: preorder[1 : mid+1]
    # 左子树的中序: inorder[: mid]
    root.left = build_tree(preorder[1 : mid + 1], inorder[: mid])
    # 右子树的前序: preorder[mid+1 :]
    # 右子树的中序: inorder[mid+1 :]
    root.right = build_tree(preorder[mid + 1 :], inorder[mid + 1 :])

    return root
```

> **优化**：`inorder.index()` 是 \(O(n)\) 的，可以提前建哈希表加速到 \(O(1)\)。

### 10.4 验证二叉搜索树（LeetCode 98）

**自顶向下传递范围约束**：

```python
def is_valid_bst(root: TreeNode | None) -> bool:
    """LeetCode 98: 验证二叉搜索树。"""

    def validate(
        node: TreeNode | None,
        low: float,
        high: float,
    ) -> bool:
        if node is None:
            return True
        if not (low < node.val < high):
            return False
        # 左子树所有值 < node.val；右子树所有值 > node.val
        return (
            validate(node.left, low, node.val)
            and validate(node.right, node.val, high)
        )

    return validate(root, float("-inf"), float("inf"))
```

### 10.5 二叉树的最近公共祖先（LeetCode 236）

这道题是面试高频题，也是递归思维的绝佳训练。

```
步骤 1 ▸ lca(root, p, q) → 在以 root 为根的树中找 p 和 q 的最近公共祖先
步骤 2 ▸ root 为 None → 返回 None
         root 就是 p 或 q → 返回 root
步骤 3 ▸ 信任 lca(root.left, p, q) 在左子树中查找
         信任 lca(root.right, p, q) 在右子树中查找
步骤 4 ▸ 如果左右都找到了 → root 就是 LCA
         如果只有左边找到 → LCA 在左边
         如果只有右边找到 → LCA 在右边
```

```python
def lowest_common_ancestor(
    root: TreeNode | None,
    p: TreeNode,
    q: TreeNode,
) -> TreeNode | None:
    """LeetCode 236: 二叉树的最近公共祖先。"""
    # 基线条件
    if root is None or root == p or root == q:
        return root

    # 信任子调用：在左右子树中分别查找
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    # 当前层逻辑
    if left and right:
        return root     # p 和 q 分别在左右子树中 → root 就是 LCA
    return left or right  # 都在同一侧
```

**为什么这段代码是正确的？**

不要试图展开递归！用"信任法"理解：

- 如果 `left` 不为 None，说明左子树中"找到了 p 或 q（或它们的 LCA）"。
- 如果 `right` 不为 None，同理。
- 如果都不为 None，说明 p 和 q 分散在当前节点的两侧 → 当前节点就是 LCA。

---

## 第十一章：非二叉树的递归

递归不仅用于二叉树，它在数组、字符串、图等各种问题中都大量使用。

### 11.1 数组：归并排序

```python
def merge_sort(arr: list[int]) -> list[int]:
    """归并排序（分治递归）。"""
    # 基线条件
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    # 分（信任子调用）
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    # 合
    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    """合并两个有序数组。"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print(merge_sort([38, 27, 43, 3, 9, 82, 10]))
# [3, 9, 10, 27, 38, 43, 82]
```

### 11.2 排列与组合（回溯经典）

#### 全排列（LeetCode 46）

```python
def permute(nums: list[int]) -> list[list[int]]:
    """LeetCode 46: 全排列。"""
    result: list[list[int]] = []

    def backtrack(path: list[int], remaining: list[int]) -> None:
        if not remaining:
            result.append(path[:])
            return
        for i, num in enumerate(remaining):
            path.append(num)
            backtrack(path, remaining[:i] + remaining[i + 1:])
            path.pop()  # 回溯：撤销选择

    backtrack([], nums)
    return result

print(permute([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

#### 子集（LeetCode 78）

```python
def subsets(nums: list[int]) -> list[list[int]]:
    """LeetCode 78: 子集。"""
    result: list[list[int]] = []

    def backtrack(start: int, path: list[int]) -> None:
        result.append(path[:])  # 每个节点都是一个有效子集
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

print(subsets([1, 2, 3]))
# [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### 11.3 字符串递归：生成有效括号（LeetCode 22）

```python
def generate_parenthesis(n: int) -> list[str]:
    """LeetCode 22: 括号生成。"""
    result: list[str] = []

    def backtrack(current: str, open_count: int, close_count: int) -> None:
        # 基线条件：生成了完整的括号串
        if len(current) == 2 * n:
            result.append(current)
            return
        # 选择 1：还能放左括号吗？
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        # 选择 2：还能放右括号吗？（右括号数量不能超过左括号）
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result

print(generate_parenthesis(3))
# ['((()))', '(()())', '(())()', '()(())', '()()()']
```

---

## 第十二章：递归思维训练清单

### 12.1 分级刷题路线

#### 第一阶段：建立信任（入门）

| 题号 | 题目 | 核心练习点 |
|------|------|-----------|
| 509 | 斐波那契数 | 最基础的递归 + 记忆化 |
| 206 | 反转链表 | 链表递归的基本功 |
| 104 | 二叉树的最大深度 | 自底向上范式 |
| 226 | 翻转二叉树 | 后序位置处理 |
| 100 | 相同的树 | 同时递归两棵树 |

#### 第二阶段：熟练运用（进阶）

| 题号 | 题目 | 核心练习点 |
|------|------|-----------|
| 101 | 对称二叉树 | 镜像递归思维 |
| 110 | 平衡二叉树 | 返回值携带多重信息 |
| 112 | 路径总和 | 自顶向下传参 |
| 98 | 验证二叉搜索树 | 范围约束传递 |
| 21 | 合并两个有序链表 | 链表递归 |

#### 第三阶段：融会贯通（高级）

| 题号 | 题目 | 核心练习点 |
|------|------|-----------|
| 236 | 最近公共祖先 | 后序位置信息收集 |
| 105 | 前序+中序构建树 | 分治法 |
| 543 | 二叉树的直径 | 返回值 ≠ 最终答案 |
| 337 | 打家劫舍 III | 树形 DP |
| 46 | 全排列 | 回溯法 |
| 78 | 子集 | 回溯法 |
| 22 | 括号生成 | 约束条件下的回溯 |
| 124 | 二叉树最大路径和 | 综合运用 |

### 12.2 每道题的练习方法

```
1. 看到题目后，不要立即写代码。
2. 在纸上写出递归三要素：
   - 基线条件是什么？
   - 递归关系是什么？（假设子问题已解决，当前层做什么？）
   - 返回值是什么？
3. 把三要素翻译为代码。
4. 用 2-3 个节点的最小用例验证。
5. 如果不对，检查你的"合同"是否在每一层都成立。
6. 做完后思考：
   - 能否用迭代实现？
   - 有没有重叠子问题？需不需要记忆化？
   - 时间 / 空间复杂度是多少？
```

---

## 总结：递归的核心心法

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   1. 不要展开递归。信任你的函数能正确处理子问题。            │
│                                                          │
│   2. 只关注当前层。你只需要回答：                           │
│      "假设子问题已解决，我还需要做什么？"                    │
│                                                          │
│   3. 永远先写基线条件。这是你的安全网。                      │
│                                                          │
│   4. 明确返回值的含义。                                    │
│      函数返回什么？这决定了你如何利用子问题的结果。            │
│                                                          │
│   5. 递归是一种分解问题的思维方式，不仅仅是一种语法。          │
│      先想清楚分解策略，代码只是翻译。                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

> **最后的建议**：递归能力不是通过阅读获得的，而是通过**反复练习**建立的。把本笔记中的每一道题，都在 LeetCode 上亲手写一遍。当你能不假思索地用"三要素框架"分析一道新题时，递归就真正属于你了。
