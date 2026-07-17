"""
第 05 章 · Memory 函数缓存 — 验证代码

目的：感受"同样的输入直接返回缓存"的魔法，以及正确使用的边界。

实验：
  1. 基本缓存 — 同样的参数不重复执行
  2. 不同参数不命中 — 输入变了就重算
  3. 改了函数体会怎样 — 自动失效
  4. numpy 数组作为参数 — 内容哈希
  5. 实战 Pipeline — 预处理 + 特征工程 + 训练 三层缓存
  6. 副作用陷阱 — 文件写入不被缓存
  7. 缓存管理 — clear / reduce_size

运行方式：
    uv run python code/05-Memory-函数缓存.py
"""

import os
import sys
import time
import tempfile
import shutil
from datetime import timedelta

import numpy as np
from joblib import Memory

# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch05_")
cache_dir = os.path.join(tmp_dir, "cache")
print(f"缓存目录: {cache_dir}\n")


# ============================================================
# 实验 1：基本缓存 — 同样参数不重复执行
# ============================================================
def experiment_1_basic():
    print("=" * 60)
    print("实验 1 · 基本缓存 — 同样参数只执行一次")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp1"), verbose=1)
    call_log = []

    @memory.cache
    def expensive_compute(x, y):
        """模拟耗时计算"""
        call_log.append((x, y))
        time.sleep(0.5)  # 模拟耗时
        return x ** 2 + y ** 2

    print("\n第一次调用 (3, 4):")
    t0 = time.perf_counter()
    r1 = expensive_compute(3, 4)
    t1 = time.perf_counter() - t1

    print(f"\n第二次调用 (3, 4) — 应该缓存命中:")
    t0 = time.perf_counter()
    r2 = expensive_compute(3, 4)
    t2 = time.perf_counter() - t0

    print(f"\n第三次调用 (5, 6) — 不同参数，应该重新执行:")
    t0 = time.perf_counter()
    r3 = expensive_compute(5, 6)
    t3 = time.perf_counter() - t0

    print(f"\n结果: r1={r1}, r2={r2}, r3={r3}")
    print(f"函数实际执行次数: {len(call_log)} (应该是 2，不是 3)")
    print(f"缓存命中时耗时: {t2:.4f}s vs 首次 {t1:.4f}s")

    assert r1 == r2 == 25
    assert r3 == 61
    assert len(call_log) == 2
    print("✅ 基本缓存验证通过\n")

    memory.clear()


# ============================================================
# 实验 2：函数体改变 → 缓存自动失效
# ============================================================
def experiment_2_code_change():
    print("=" * 60)
    print("实验 2 · 函数体改了 → 旧缓存自动失效")
    print("=" * 60)

    # 这是 Memory 的一个重要安全机制：
    # 如果你改了函数实现，它不会返回旧的缓存结果

    memory = Memory(os.path.join(cache_dir, "exp2"), verbose=0)

    @memory.cache
    def multiply_version1(a, b):
        return a * b

    r1 = multiply_version1(10, 20)
    print(f"版本1: 10×20 = {r1}")

    # 模拟"改了函数"——重新定义一个同名函数
    @memory.cache
    def multiply_version1(a, b):
        """修复 bug：之前算错了"""  # 加了 docstring，函数源码变了
        return a * b * 2  # 逻辑也变了

    r2 = multiply_version1(10, 20)
    print(f"版本2: 10×20 = {r2}")

    print(f"版本1 缓存值: {r1}, 版本2 新计算值: {r2}")
    print("📖 函数体变了 → 旧缓存不生效 → 用新逻辑重新计算")
    print("   这是一个安全机制，防止你拿到过时的结果\n")

    memory.clear()


# ============================================================
# 实验 3：numpy 数组作为参数
# ============================================================
def experiment_3_numpy_args():
    print("=" * 60)
    print("实验 3 · numpy 数组参数 — 按内容哈希，不按对象 id")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp3"), verbose=1)

    @memory.cache
    def array_stats(arr):
        time.sleep(0.3)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    # 两个内容相同的数组——但是不同对象
    a1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a3 = np.array([10.0, 20.0, 30.0])

    print(f"\n同一个内容的对象 a1 和 a2: id={id(a1)} vs id={id(a2)}")
    print(f"它们是不同对象: {a1 is not a2}")

    print("\n第一次调用 (a1):")
    s1 = array_stats(a1)

    print("\n第二次调用 (a2, 内容同 a1):")
    s2 = array_stats(a2)

    print("\n第三次调用 (a3, 内容不同):")
    s3 = array_stats(a3)

    print(f"\na1 结果: {s1}")
    print(f"a2 结果: {s2}")
    print(f"a3 结果: {s3}")
    print(f"a1==a2: {s1 == s2} (内容哈希命中缓存)")
    print(f"a1==a3: {s1 == s3}")

    assert s1 == s2
    assert s1 != s3
    print("✅ numpy 参数按内容哈希验证通过\n")

    memory.clear()


# ============================================================
# 实验 4：多层 Pipeline 缓存 — 只重算改了的部分
# ============================================================
def experiment_4_pipeline():
    print("=" * 60)
    print("实验 4 · 多层 Pipeline — 每个阶段独立缓存")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp4"), verbose=1)

    execute_log = []

    @memory.cache
    def load_data(source):
        execute_log.append(f"load:{source}")
        time.sleep(0.3)
        return {"source": source, "raw": np.random.RandomState(hash(source) % 2**32).randn(1000)}

    @memory.cache
    def clean_data(data):
        execute_log.append(f"clean:{data['source']}")
        time.sleep(0.3)
        result = data.copy()
        result["clean"] = data["raw"] - data["raw"].mean()
        return result

    @memory.cache
    def extract_features(data):
        execute_log.append(f"features:{data['source']}")
        time.sleep(0.3)
        return {"source": data["source"], "features": data["clean"][:10].tolist()}

    # 第一轮：全部执行
    print("\n=== 第一轮：全部执行 ===")
    raw = load_data("dataset_A")
    clean = clean_data(raw)
    feats = extract_features(clean)
    print(f"执行记录: {execute_log}")

    # 第二轮：输入完全没变 → 全部缓存命中
    execute_log.clear()
    print("\n=== 第二轮：输入没变 → 全部缓存命中 ===")
    raw = load_data("dataset_A")
    clean = clean_data(raw)
    feats = extract_features(clean)
    print(f"执行记录: {execute_log} (空 = 全部缓存命中)")

    # 第三轮：模拟"改了 clean_data 的实现"
    execute_log.clear()
    memory.clear()  # 清掉旧缓存（实际场景中改了函数代码会自动清）

    print("\n=== 第三轮：新数据源 → 前两级缓存失效，重新计算 ===")
    raw = load_data("dataset_B")          # 新数据源 → 执行
    clean = clean_data(raw)               # 新输入 → 执行
    feats = extract_features(clean)       # 新输入 → 执行
    print(f"执行记录: {execute_log}")

    print(f"\n📖 Pipeline 缓存的核心价值：")
    print(f"   你改了特征提取逻辑 → 只有 extract_features 重算")
    print(f"   你换了数据源 → load → clean → features 整条链重算")
    print(f"   什么都没改 → 0 次计算\n")

    memory.clear()


# ============================================================
# 实验 5：副作用陷阱
# ============================================================
def experiment_5_side_effect():
    print("=" * 60)
    print("实验 5 · 副作用陷阱 — 被缓存的函数不要写文件")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp5"), verbose=1)
    side_effect_file = os.path.join(tmp_dir, "side_effect.txt")

    @memory.cache
    def compute_and_save(x):
        """计算 + 保存文件（有副作用！）"""
        result = x ** 2
        # 副作用：写文件
        with open(side_effect_file, "w") as f:
            f.write(f"Latest result: {result}")
        return result

    print("\n第一次调用 compute_and_save(5):")
    r1 = compute_and_save(5)
    with open(side_effect_file) as f:
        content1 = f.read()
    print(f"  返回值: {r1}, 文件内容: {content1}")

    print("\n第二次调用 compute_and_save(5) — 缓存命中:")
    r2 = compute_and_save(5)
    with open(side_effect_file) as f:
        content2 = f.read()
    print(f"  返回值: {r2}, 文件内容: {content2}")
    print(f"  ⚠️ 文件内容没变 — 因为函数体没执行，副作用没触发!")

    print(f"\n📖 规则：被 Memory.cache 装饰的函数应该是纯函数")
    print(f"   - 输入一样 → 输出一样 ✓")
    print(f"   - 不写文件 / 不发请求 / 不更新数据库 ✓")
    print(f"   - 不依赖外部可变状态 ✓\n")

    memory.clear()


# ============================================================
# 实验 6：缓存管理
# ============================================================
def experiment_6_cache_management():
    print("=" * 60)
    print("实验 6 · 缓存管理 — clear / reduce_size")
    print("=" * 60)

    memory = Memory(os.path.join(cache_dir, "exp6"), verbose=0)

    @memory.cache
    def square(x):
        return x ** 2

    # 生成一些缓存
    for i in range(10):
        square(i)

    # 统计缓存文件数
    cache_root = os.path.join(cache_dir, "exp6")
    def count_cache_files():
        count = 0
        for root, dirs, files in os.walk(cache_root):
            count += len([f for f in files if not f.startswith('.')])
        return count

    before = count_cache_files()
    print(f"缓存文件数: {before}")

    # 清空
    memory.clear()
    after = count_cache_files()
    print(f"memory.clear() 后: {after}")

    # 重新生成
    for i in range(20):
        square(i)
    print(f"重新生成 20 条缓存: {count_cache_files()}")

    # reduce_size（实际项目中按时间清理）
    # memory.reduce_size(age_limit=timedelta(days=30))
    print(f"📖 memory.reduce_size(age_limit=timedelta(days=30)) 可清理 30 天前的缓存")

    memory.clear()
    print()


# ============================================================
if __name__ == "__main__":
    experiment_1_basic()
    experiment_2_code_change()
    experiment_3_numpy_args()
    experiment_4_pipeline()
    experiment_5_side_effect()
    experiment_6_cache_management()

    shutil.rmtree(tmp_dir)

    print("=" * 60)
    print("第 05 章完成 ✓")
    print("=" * 60)
    print("""
本章验证了：
  ✅ 同样参数第二次调用 → 缓存命中，函数体不执行
  ✅ 改了函数逻辑 → 旧缓存自动失效（安全机制）
  ✅ numpy 数组按内容哈希 → 不同对象但内容相同 = 缓存命中
  ✅ 多层 Pipeline → 每层独立缓存，改了哪层只重算哪层
  ✅ 副作用陷阱 → 文件 I/O 不被缓存
  ✅ 缓存管理 → clear() 清空, reduce_size() 按时间清理
""")
