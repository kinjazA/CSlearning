"""
第 02 章 · dump & load 序列化 — 验证代码

目的：建立肌肉记忆 — 存模型、读模型、压缩、mmap、避坑。

实验：
  0. pickle 基础 — dump/load/dumps/loads + protocol
  1. 基本读写 + pickle vs joblib 速度对比
  2. 压缩 vs 不压缩，三种典型场景
  3. mmap 加载大文件 — 加载时间 + 只读验证
  4. sklearn Pipeline roundtrip
  5. 六个常见坑的演示

运行方式：
    uv run python code/02-dump-load-序列化.py
"""

import os
import sys
import time
import pickle
import tempfile
import shutil
from io import BytesIO
from datetime import date

import numpy as np
import joblib

# ============================================================
tmp_dir = tempfile.mkdtemp(prefix="joblib_ch02_")
print(f"Python: {sys.version.split()[0]}, joblib: {joblib.__version__}\n")


# ============================================================
# 实验 0：pickle 基础 — dump/load/dumps/loads + protocol
# ============================================================
def experiment_0_pickle_basics():
    print("=" * 60)
    print("实验 0 · pickle 基础 — Python 自带的序列化")
    print("=" * 60)

    # ── 0a. 文件操作：dump + load ──
    print("\n── 0a. 文件操作：pickle.dump / pickle.load ──")
    data = {"model": "RF", "accuracy": 0.92, "params": [100, 10]}

    path = os.path.join(tmp_dir, "demo.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  保存: {os.path.getsize(path)} bytes")

    with open(path, "rb") as f:
        loaded = pickle.load(f)
    print(f"  加载: {loaded}")
    assert data == loaded
    print(f"  ✅ roundtrip 一致")

    # ── 0b. 内存操作：dumps + loads ──
    print("\n── 0b. 内存操作：pickle.dumps / pickle.loads ──")
    arr = np.arange(100, dtype=np.float64)
    raw_bytes = pickle.dumps(arr)
    print(f"  dumps() 返回类型: {type(raw_bytes).__name__}, 大小: {len(raw_bytes)} bytes")

    restored = pickle.loads(raw_bytes)
    print(f"  loads() 还原: shape={restored.shape}, dtype={restored.dtype}")
    assert np.array_equal(arr, restored)
    print(f"  ✅ dumps/loads roundtrip 一致")

    # ── 0c. protocol 版本对比 ──
    print("\n── 0c. protocol 版本对比 ──")
    big_arr = np.random.randn(100_000).astype(np.float64)  # ~800 KB

    protocols = [
        (0, "ASCII 文本，人类可读"),
        (1, "旧版二进制"),
        (3, "Python 3 默认"),
        (4, "Python 3.4+ 推荐"),
        (pickle.HIGHEST_PROTOCOL, f"最高协议 (={pickle.HIGHEST_PROTOCOL})"),
    ]

    print(f"  {'protocol':<12} {'大小':>10} {'写入':>10} {'读取':>10} 说明")
    print(f"  {'-'*60}")
    for proto, desc in protocols:
        t0 = time.perf_counter()
        b = pickle.dumps(big_arr, protocol=proto)
        t_dump = time.perf_counter() - t0

        t0 = time.perf_counter()
        pickle.loads(b)
        t_load = time.perf_counter() - t0

        print(f"  {proto:<12} {len(b)/1024:>8.1f}KB {t_dump:>9.4f}s {t_load:>9.4f}s  {desc}")

    print(f"\n  📖 protocol 越高，体积越小、速度越快")
    print(f"  📖 日常用 pickle.HIGHEST_PROTOCOL（自动选最高）")
    print(f"  📖 protocol=0 体积巨大（约其他协议的 3-4×），仅用于调试")

    # ── 0d. pickle 适用场景速览 ──
    print("\n── 0d. pickle vs joblib 适用场景 ──")
    small_obj = {"name": "test", "score": 0.95}

    # pickle 处理小对象
    t0 = time.perf_counter()
    pk = pickle.dumps(small_obj)
    t_pickle = time.perf_counter() - t0

    # joblib 处理小对象
    t0 = time.perf_counter()
    jl = joblib.dumps(small_obj)
    t_joblib = time.perf_counter() - t0

    print(f"  小 dict: pickle={t_pickle*1e6:.0f}μs, joblib={t_joblib*1e6:.0f}μs")
    print(f"  → 简单小对象用 pickle（零依赖，够快）")

    big_arr = np.random.randn(500_000).astype(np.float64)
    t0 = time.perf_counter()
    pk = pickle.dumps(big_arr)
    t_pickle = time.perf_counter() - t0

    t0 = time.perf_counter()
    jl = joblib.dump(big_arr, os.path.join(tmp_dir, "_exp0.joblib"), compress=0)
    t_joblib = time.perf_counter() - t0

    print(f"  大数组: pickle={t_pickle:.3f}s, joblib={t_joblib:.3f}s")
    print(f"  → 大数组/模型用 joblib（快 2-5×）")
    print(f"  ✅ pickle 基础实验完成\n")


# ============================================================
# 实验 1：基本读写 + pickle vs joblib
# ============================================================
def experiment_1_basic():
    print("=" * 60)
    print("实验 1 · 基本读写 + pickle vs joblib")
    print("=" * 60)

    # 1a. 最简用法
    arr = np.arange(1000, dtype=np.float64)
    joblib.dump(arr, os.path.join(tmp_dir, "demo.joblib"))
    loaded = joblib.load(os.path.join(tmp_dir, "demo.joblib"))
    assert np.array_equal(arr, loaded)
    print("✅ 基本读写正确")

    # 1b. BytesIO — 存内存不落盘
    buf = BytesIO()
    joblib.dump(arr, buf, compress=3)
    buf.seek(0)
    restored = joblib.load(buf)
    assert np.array_equal(arr, restored)
    print(f"✅ BytesIO 内存读写正确 ({buf.getbuffer().nbytes} bytes)")

    # 1c. pickle vs joblib 速度对比
    print(f"\n{'规模':<12} {'pickle写':>10} {'joblib写':>10} {'加速比':>8} | {'体积比':>8}")
    print("-" * 54)
    for n, label in [(10_000, "1万"), (100_000, "10万"), (1_000_000, "100万")]:
        a = np.random.randn(n).astype(np.float64)
        # pickle
        pk = os.path.join(tmp_dir, f"pk_{n}.pkl")
        t0 = time.perf_counter()
        with open(pk, "wb") as f: pickle.dump(a, f)
        tp = time.perf_counter() - t0
        sp = os.path.getsize(pk)
        # joblib
        jl = os.path.join(tmp_dir, f"jl_{n}.joblib")
        t0 = time.perf_counter()
        joblib.dump(a, jl, compress=0)
        tj = time.perf_counter() - t0
        sj = os.path.getsize(jl)
        print(f"{label:<12} {tp:>9.4f}s {tj:>9.4f}s {tp/tj:>7.1f}x | {sj/sp*100:>7.1f}%")

    print("📖 数组越大，joblib 优势越明显")


# ============================================================
# 实验 2：压缩 — 三种典型场景
# ============================================================
def experiment_2_compress():
    print(f"\n{'='*60}")
    print("实验 2 · 压缩 — 三种典型场景")
    print("=" * 60)

    arr = np.random.randn(1_000_000).astype(np.float64)
    raw_mb = arr.nbytes / 1024**2

    configs = [
        ("日常开发 (compress=3)",        3),
        ("追求速度 (lz4-0)",             ("lz4", 0)),
        ("最小体积 (lzma-9)",            ("lzma", 9)),
    ]
    try:
        import zstandard
        configs.insert(2, ("最佳平衡 (zstd-3)", ("zstd", 3)))
    except ImportError:
        pass
    try:
        import lz4
    except ImportError:
        configs = [c for c in configs if "lz4" not in str(c[1])]

    print(f"原始体积: {raw_mb:.1f} MB\n")
    print(f"{'场景':<30} {'写入':>8} {'读取':>8} {'文件':>10} {'压缩率':>8}")
    print("-" * 68)

    for label, comp in configs:
        path = os.path.join(tmp_dir, f"scene_{label[:10]}.joblib")
        t0 = time.perf_counter()
        joblib.dump(arr, path, compress=comp)
        tw = time.perf_counter() - t0
        t0 = time.perf_counter()
        joblib.load(path)
        tr = time.perf_counter() - t0
        sz = os.path.getsize(path) / 1024**2
        print(f"{label:<30} {tw:>7.3f}s {tr:>7.3f}s {sz:>8.1f}MB {sz/raw_mb*100:>7.1f}%")

    print("\n📖 日常用 compress=3，速度优先用 lz4，体积优先用 lzma")


# ============================================================
# 实验 3：mmap 大文件加载
# ============================================================
def experiment_3_mmap():
    print(f"\n{'='*60}")
    print("实验 3 · mmap — 大文件的正确打开方式")
    print("=" * 60)

    n = 15_000_000  # ~120 MB
    print(f"创建 {n:,} 元素数组 ({n*8/1024**2:.0f} MB)...")
    arr = np.random.randn(n).astype(np.float64)

    path = os.path.join(tmp_dir, "big_mmap.joblib")
    joblib.dump(arr, path, compress=0)
    del arr

    # 普通加载
    print("\n普通加载 (mmap_mode=None)...")
    t0 = time.perf_counter()
    full = joblib.load(path, mmap_mode=None)
    t_full = time.perf_counter() - t0
    print(f"  耗时: {t_full:.2f}s, 全部读入内存")

    # mmap 加载
    print("\nmmap 加载 (mmap_mode='r')...")
    t0 = time.perf_counter()
    mm = joblib.load(path, mmap_mode='r')
    t_mmap = time.perf_counter() - t0
    print(f"  耗时: {t_mmap:.3f}s (几乎瞬间 — 只建了映射)")
    print(f"  加速: {t_full/t_mmap:.0f}×")

    # 按需读取
    print("\n按需访问验证...")
    t0 = time.perf_counter()
    _ = mm[:1000].copy()       # 首次访问 → page fault
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = mm[:1000].copy()       # 再次访问 → 已在页缓存
    t2 = time.perf_counter() - t0
    print(f"  首次访问前1000个: {t1*1000:.2f}ms (触发 page fault)")
    print(f"  再次访问前1000个: {t2*1000:.2f}ms (已在缓存)")

    # 只读限制
    try:
        mm[0] = 999.0
    except (ValueError, OSError) as e:
        print(f"\n  mmap='r' 是只读的: {type(e).__name__}")

    del mm, full
    print("📖 大文件用 mmap_mode='r'，加载瞬间完成，用到哪读到哪")


# ============================================================
# 实验 4：sklearn Pipeline roundtrip
# ============================================================
def experiment_4_sklearn():
    print(f"\n{'='*60}")
    print("实验 4 · sklearn Pipeline roundtrip")
    print("=" * 60)

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=3000, n_features=20, n_classes=3, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ])
    pipe.fit(X, y)

    score_before = pipe.score(X, y)

    # 保存 — 版本化命名
    today = date.today().isoformat()
    model_path = os.path.join(tmp_dir, f"pipeline_v1_{today}.joblib")
    joblib.dump(pipe, model_path, compress=3)
    print(f"已保存: {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")

    # 加载
    loaded = joblib.load(model_path)
    score_after = loaded.score(X, y)

    # 验证
    assert score_before == score_after
    pred_before = pipe.predict(X[:5])
    pred_after = loaded.predict(X[:5])
    assert np.array_equal(pred_before, pred_after)

    print(f"准确率: 保存前={score_before:.4f}, 加载后={score_after:.4f}")
    print(f"单条预测: {pred_before[0]} — 完全一致")

    # 同时保存元信息
    meta = {"version": "1.0", "date": today, "accuracy": score_before}
    joblib.dump(meta, os.path.join(tmp_dir, f"pipeline_v1_{today}_meta.joblib"))
    print(f"✅ 模型 + 元信息 全部保存")


# ============================================================
# 实验 4b：元信息模式 — load-before-load
# ============================================================
def experiment_4b_meta_pattern():
    print(f"\n{'='*60}")
    print("实验 4b · 元信息模式 — 生产环境的标配")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    feature_names = [f"feature_{i}" for i in range(20)]

    # ── 模拟训练了多个版本 ──
    print("\n── 模拟三次实验，每次不同的特征和参数 ──")
    experiments = [
        {"features": feature_names[:10], "n_est": 100, "max_d": 10,  "ver": "1.0"},
        {"features": feature_names[:15], "n_est": 200, "max_d": 15,  "ver": "2.0"},
        {"features": feature_names,      "n_est": 300, "max_d": None, "ver": "3.0"},
    ]

    model_dir = os.path.join(tmp_dir, "model_registry")
    os.makedirs(model_dir, exist_ok=True)

    today = date.today().isoformat()

    for exp in experiments:
        # 用指定特征训练
        col_idx = [int(f.split("_")[1]) for f in exp["features"]]
        model = RandomForestClassifier(
            n_estimators=exp["n_est"], max_depth=exp["max_d"],
            n_jobs=1, random_state=42
        )
        model.fit(X[:, col_idx], y)
        acc = float(cross_val_score(model, X[:, col_idx], y, cv=3).mean())

        # 保存模型
        model_name = f"churn_rf_v{exp['ver']}_{today}"
        joblib.dump(model, os.path.join(model_dir, f"{model_name}.joblib"), compress=3)

        # 保存完整元信息
        import sklearn
        meta = {
            "model_name": "customer_churn_rf",
            "version": exp["ver"],
            "train_date": today,
            "trainer": "zhangsan",
            "metrics": {"accuracy": round(acc, 4)},
            "feature_names": exp["features"],
            "feature_count": len(exp["features"]),
            "hyperparams": {"n_estimators": exp["n_est"], "max_depth": exp["max_d"]},
            "sklearn_version": sklearn.__version__,
            "python_version": sys.version.split()[0],
            "training_samples": len(y),
            "notes": f"v{exp['ver']}: {len(exp['features'])} 特征, 精度={acc:.3f}",
        }
        joblib.dump(meta, os.path.join(model_dir, f"{model_name}_meta.joblib"))
        print(f"  保存: v{exp['ver']} | {len(exp['features'])}特征 | acc={acc:.3f}")

    # ── 场景1：浏览所有模型（不加载模型本身）──
    print(f"\n── 场景1：浏览模型目录（只读 meta，不碰大文件）──")
    import glob
    for meta_path in sorted(glob.glob(os.path.join(model_dir, "*_meta.joblib"))):
        m = joblib.load(meta_path)
        print(f"  {m['version']} | {m['train_date']} | "
              f"acc={m['metrics']['accuracy']:.3f} | "
              f"{m['feature_count']}特征 | {m['notes']}")

    # ── 场景2：自动选最优版本 ──
    print(f"\n── 场景2：自动选出精度最高的版本 ──")
    best_meta = None
    best_acc = 0
    for meta_path in glob.glob(os.path.join(model_dir, "*_meta.joblib")):
        m = joblib.load(meta_path)
        if m["metrics"]["accuracy"] > best_acc:
            best_acc = m["metrics"]["accuracy"]
            best_meta = m
            best_meta_path = meta_path

    print(f"  最佳版本: v{best_meta['version']}, acc={best_acc:.3f}")
    # 只加载这一个模型
    model_path = best_meta_path.replace("_meta.joblib", ".joblib")
    best_model = joblib.load(model_path)
    print(f"  模型类型: {type(best_model).__name__}, 加载成功 ✅")

    # ── 场景3：特征校验 ──
    print(f"\n── 场景3：部署前校验特征是否匹配 ──")
    # 模拟：当前生产环境的数据列
    current_columns = {"feature_0", "feature_1", "feature_2", "feature_5", "user_id"}

    required = set(best_meta["feature_names"])
    missing = required - current_columns
    extra = current_columns - required

    if missing:
        print(f"  ❌ 数据缺失特征: {missing} → 拒绝部署")
    else:
        print(f"  ✅ 所需特征齐全")
    if extra - {"user_id"}:
        print(f"  ⚠️ 数据多余特征（将被忽略）: {extra - {'user_id'}}")

    print(f"\n  📖 核心：meta 只有几 KB，加载瞬间完成，不碰模型文件")
    print(f"  📖 拿到 meta → 判断 → 决定是否加载大模型 → 避免浪费内存和时间")


# ============================================================
# 实验 5：六个常见坑
# ============================================================
def experiment_5_pitfalls():
    print(f"\n{'='*60}")
    print("实验 5 · 常见坑演示")
    print("=" * 60)

    # 坑 1：dict 打包 → 无法选择性 mmap
    print("\n坑 1 — 全部打包 vs 分别保存:")
    X = np.random.randn(100_000, 50)    # ~40 MB
    y = np.random.randn(100_000)         # ~0.8 MB
    bundle = os.path.join(tmp_dir, "bundle.joblib")
    joblib.dump({"X": X, "y": y}, bundle, compress=3)
    print(f"  打包保存: {os.path.getsize(bundle)/1024**2:.1f} MB (加载时必须全部读入)")
    xp = os.path.join(tmp_dir, "X_separate.joblib")
    yp = os.path.join(tmp_dir, "y_separate.joblib")
    joblib.dump(X, xp, compress=3)
    joblib.dump(y, yp, compress=3)
    X_m = joblib.load(xp, mmap_mode='r')
    print(f"  分别保存: X={os.path.getsize(xp)/1024**2:.1f}MB, y={os.path.getsize(yp)/1024:.1f}MB")
    print(f"  可以用 mmap 单独加载 X: {isinstance(X_m, np.memmap)}")

    # 坑 2：文件覆盖
    print("\n坑 2 — 不版本化导致覆盖:")
    for i in range(3):
        joblib.dump(np.array([i]), os.path.join(tmp_dir, "model.joblib"))
    print(f"  覆盖3次后: {joblib.load(os.path.join(tmp_dir, 'model.joblib'))} (只剩最后一个)")

    # 版本化
    for i in range(1, 4):
        joblib.dump(np.array([i]), os.path.join(tmp_dir, f"model_v{i}.joblib"))
    print(f"  版本化后: v1={joblib.load(os.path.join(tmp_dir, 'model_v1.joblib'))}, "
          f"v2={joblib.load(os.path.join(tmp_dir, 'model_v2.joblib'))}, "
          f"v3={joblib.load(os.path.join(tmp_dir, 'model_v3.joblib'))}")

    # 坑 3：压缩 + mmap 互斥
    print("\n坑 3 — 压缩后 mmap 失效:")
    c0 = os.path.join(tmp_dir, "mmap_test_c0.joblib")
    c3 = os.path.join(tmp_dir, "mmap_test_c3.joblib")
    a = np.random.randn(1_000_000)
    joblib.dump(a, c0, compress=0)
    joblib.dump(a, c3, compress=3)
    print(f"  compress=0 → mmap: {isinstance(joblib.load(c0, mmap_mode='r'), np.memmap)}")
    print(f"  compress=3 → mmap: {isinstance(joblib.load(c3, mmap_mode='r'), np.memmap)} (静默忽略!)")

    # 坑 4：mmap 只读
    print("\n坑 4 — mmap='r' 只读:")
    ld = joblib.load(c0, mmap_mode='r')
    try:
        ld[0] = 999.0
    except (ValueError, OSError) as e:
        print(f"  修改被拒绝: {type(e).__name__} → 用 np.array(data) 先复制到内存")

    # 坑 5：load 可执行代码
    print("\n坑 5 — 安全提醒:")
    print("  load() 可以执行任意代码，只加载信任来源的文件")

    # 坑 6：sklearn 版本
    print("\n坑 6 — sklearn 版本兼容:")
    print("  训练和部署用同一个 sklearn 版本")

    del X, y, X_m, a, ld

    print(f"\n{'='*60}")
    print("6 个坑 = 全部演示完毕")


# ============================================================
if __name__ == "__main__":
    experiment_0_pickle_basics()
    experiment_1_basic()
    experiment_2_compress()
    experiment_3_mmap()
    experiment_4_sklearn()
    experiment_4b_meta_pattern()
    experiment_5_pitfalls()

    shutil.rmtree(tmp_dir)

    print(f"\n{'='*60}")
    print("第 02 章完成 ✓")
    print("=" * 60)
    print("""
本章验证了：
  ✅ pickle 基础 — dump/load/dumps/loads + protocol 版本对比
  ✅ pickle vs joblib — 小对象用 pickle，大数组/模型用 joblib
  ✅ dump/load 基本用法 + BytesIO 内存读写
  ✅ 三种典型压缩场景的速度/体积对比
  ✅ mmap_mode='r' 大文件瞬间加载 + 按需分页
  ✅ sklearn Pipeline 完整 roundtrip
  ✅ 元信息模式 — load-before-load（浏览、选优、校验、部署门禁）
  ✅ 6 个常见坑 = 打包存、文件覆盖、压缩+mmap互斥、只读、安全、版本兼容
""")
