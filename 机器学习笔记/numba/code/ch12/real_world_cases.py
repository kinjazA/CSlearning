"""
第十二章：实战案例 —— 配套代码
===================================
学习目标：
  1. 机器学习特征工程加速
  2. 蒙特卡罗模拟（期权定价、π 估算）
  3. 数值计算（梯度下降、ODE 求解、数值积分）
  4. 综合案例（粒子滤波）
"""
import numpy as np
import time
from numba import njit, prange

# ══════════════════════════════════════════════
# 案例 1: 距离矩阵计算
# ══════════════════════════════════════════════

@njit(parallel=True)
def distance_matrix_parallel(X):
    """并行计算欧氏距离矩阵"""
    n, d = X.shape
    D = np.zeros((n, n))
    for i in prange(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            D[i, j] = dist
            D[j, i] = dist
    return D


def demo_distance_matrix():
    print("=" * 55)
    print("案例 1: 距离矩阵计算（KNN/聚类）")
    print("=" * 55)

    X = np.random.randn(500, 20).astype(np.float64)

    # 热身
    distance_matrix_parallel(X[:10])

    t0 = time.perf_counter()
    D = distance_matrix_parallel(X)
    t = time.perf_counter() - t0

    print(f"  数据: {X.shape[0]} 样本 × {X.shape[1]} 特征")
    print(f"  距离矩阵: {D.shape}")
    print(f"  计算时间: {t:.4f}s")
    print(f"  示例距离: {D[0, 1]:.4f}")


# ══════════════════════════════════════════════
# 案例 2: 滑动窗口特征
# ══════════════════════════════════════════════

@njit
def rolling_features(data, window):
    """滑动窗口统计特征"""
    n = len(data)
    means = np.empty(n)
    stds = np.empty(n)
    maxs = np.empty(n)
    mins = np.empty(n)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]

        means[i] = np.mean(window_data)
        stds[i] = np.std(window_data)
        maxs[i] = np.max(window_data)
        mins[i] = np.min(window_data)

    return means, stds, maxs, mins


def demo_rolling_features():
    print(f"\n{'='*55}")
    print("案例 2: 滑动窗口特征（时序数据）")
    print(f"{'='*55}")

    np.random.seed(42)
    data = np.cumsum(np.random.randn(10000))

    # 热身
    rolling_features(data[:100], 20)

    t0 = time.perf_counter()
    mean, std, max_val, min_val = rolling_features(data, window=50)
    t = time.perf_counter() - t0

    print(f"  数据长度: {len(data)}")
    print(f"  窗口大小: 50")
    print(f"  计算时间: {t:.4f}s")
    print(f"  示例 (t=100): mean={mean[100]:.2f}, std={std[100]:.2f}")


# ══════════════════════════════════════════════
# 案例 3: K-Means 聚类核心
# ══════════════════════════════════════════════

@njit(parallel=True)
def assign_clusters(X, centroids):
    """为每个样本分配最近的簇"""
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]
    n_features = X.shape[1]
    labels = np.empty(n_samples, dtype=np.int32)

    for i in prange(n_samples):
        min_dist = np.inf
        best_cluster = 0

        for k in range(n_clusters):
            dist = 0.0
            for j in range(n_features):
                diff = X[i, j] - centroids[k, j]
                dist += diff * diff

            if dist < min_dist:
                min_dist = dist
                best_cluster = k

        labels[i] = best_cluster

    return labels


@njit
def update_centroids(X, labels, n_clusters):
    """更新簇中心"""
    n_samples, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features))
    counts = np.zeros(n_clusters)

    for i in range(n_samples):
        label = labels[i]
        counts[label] += 1
        for j in range(n_features):
            centroids[label, j] += X[i, j]

    for k in range(n_clusters):
        if counts[k] > 0:
            for j in range(n_features):
                centroids[k, j] /= counts[k]

    return centroids


def demo_kmeans():
    print(f"\n{'='*55}")
    print("案例 3: K-Means 聚类加速")
    print(f"{'='*55}")

    np.random.seed(42)
    X = np.random.randn(5000, 10).astype(np.float64)
    n_clusters = 5
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]

    # 热身
    assign_clusters(X[:100], centroids)

    t0 = time.perf_counter()
    labels = assign_clusters(X, centroids)
    centroids_new = update_centroids(X, labels, n_clusters)
    t = time.perf_counter() - t0

    print(f"  数据: {X.shape[0]} 样本 × {X.shape[1]} 特征")
    print(f"  簇数: {n_clusters}")
    print(f"  一次迭代时间: {t:.4f}s")
    print(f"  各簇样本数: {np.bincount(labels)}")


# ══════════════════════════════════════════════
# 案例 4: 期权定价（Monte Carlo）
# ══════════════════════════════════════════════

@njit(parallel=True)
def european_call_monte_carlo(S0, K, T, r, sigma, n_simulations):
    """欧式看涨期权 Monte Carlo 定价"""
    payoffs = np.empty(n_simulations)

    for i in prange(n_simulations):
        Z = np.random.randn()
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoffs[i] = max(ST - K, 0.0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


def demo_option_pricing():
    print(f"\n{'='*55}")
    print("案例 4: 期权定价（Monte Carlo）")
    print(f"{'='*55}")

    # 热身
    european_call_monte_carlo(100, 105, 1.0, 0.05, 0.2, 1000)

    t0 = time.perf_counter()
    price = european_call_monte_carlo(
        S0=100,      # 当前股价
        K=105,       # 行权价
        T=1.0,       # 1年到期
        r=0.05,      # 5% 无风险利率
        sigma=0.2,   # 20% 波动率
        n_simulations=1_000_000
    )
    t = time.perf_counter() - t0

    print(f"  模拟次数: 1,000,000")
    print(f"  计算时间: {t:.4f}s")
    print(f"  期权价格: ${price:.4f}")


# ══════════════════════════════════════════════
# 案例 5: π 值估算
# ══════════════════════════════════════════════

@njit(parallel=True)
def estimate_pi_monte_carlo(n_samples):
    """用单位圆内点的比例估算 π"""
    count_inside = 0

    for i in prange(n_samples):
        x = np.random.rand()
        y = np.random.rand()

        if x*x + y*y <= 1.0:
            count_inside += 1

    return 4.0 * count_inside / n_samples


def demo_pi_estimation():
    print(f"\n{'='*55}")
    print("案例 5: π 值估算（Monte Carlo）")
    print(f"{'='*55}")

    # 热身
    estimate_pi_monte_carlo(1000)

    t0 = time.perf_counter()
    pi_estimate = estimate_pi_monte_carlo(5_000_000)
    t = time.perf_counter() - t0

    print(f"  采样点数: 5,000,000")
    print(f"  计算时间: {t:.4f}s")
    print(f"  π 估算值: {pi_estimate:.6f}")
    print(f"  真实 π:   {np.pi:.6f}")
    print(f"  误差:     {abs(pi_estimate - np.pi):.6f}")


# ══════════════════════════════════════════════
# 案例 6: 梯度下降优化
# ══════════════════════════════════════════════

@njit
def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """梯度下降优化"""
    x = x0.copy()
    history = []

    for iter in range(max_iter):
        grad = grad_f(x)
        x_new = x - lr * grad

        # 检查收敛
        if np.linalg.norm(x_new - x) < tol:
            return x_new, iter + 1

        x = x_new

    return x, max_iter


@njit
def quadratic(x):
    """目标函数: f(x) = (x[0]-3)^2 + (x[1]-2)^2"""
    return (x[0] - 3)**2 + (x[1] - 2)**2


@njit
def quadratic_grad(x):
    """梯度"""
    return np.array([2*(x[0] - 3), 2*(x[1] - 2)])


def demo_gradient_descent():
    print(f"\n{'='*55}")
    print("案例 6: 梯度下降优化")
    print(f"{'='*55}")

    x0 = np.array([0.0, 0.0])

    # 热身
    gradient_descent(quadratic, quadratic_grad, x0, max_iter=10)

    t0 = time.perf_counter()
    x_opt, n_iter = gradient_descent(quadratic, quadratic_grad, x0, lr=0.1)
    t = time.perf_counter() - t0

    print(f"  目标函数: f(x,y) = (x-3)² + (y-2)²")
    print(f"  初始点:   {x0}")
    print(f"  最优解:   {x_opt}  (真实最优: [3, 2])")
    print(f"  迭代次数: {n_iter}")
    print(f"  计算时间: {t:.4f}s")
    print(f"  最优值:   {quadratic(x_opt):.8f}")


# ══════════════════════════════════════════════
# 案例 7: 数值积分（梯形法则）
# ══════════════════════════════════════════════

@njit
def trapezoidal_rule(f, a, b, n):
    """梯形法则计算积分"""
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        x = a + i * h
        integral += f(x)

    integral *= h
    return integral


@njit
def sine(x):
    return np.sin(x)


def demo_numerical_integration():
    print(f"\n{'='*55}")
    print("案例 7: 数值积分")
    print(f"{'='*55}")

    # 热身
    trapezoidal_rule(sine, 0, np.pi, 100)

    t0 = time.perf_counter()
    result = trapezoidal_rule(sine, 0, np.pi, n=100000)
    t = time.perf_counter() - t0

    print(f"  积分: ∫(0 to π) sin(x) dx")
    print(f"  分割数: 100,000")
    print(f"  计算时间: {t:.4f}s")
    print(f"  数值解: {result:.6f}")
    print(f"  真实值: 2.000000")
    print(f"  误差:   {abs(result - 2.0):.8f}")


# ══════════════════════════════════════════════
# 案例 8: 多 Seed 似然加权粒子滤波
# ══════════════════════════════════════════════

@njit
def systematic_resample(weights):
    """系统重采样（低方差）"""
    N = len(weights)
    positions = (np.arange(N) + np.random.rand()) / N

    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(N, dtype=np.int32)

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


@njit
def effective_sample_size(weights):
    """计算有效粒子数"""
    return 1.0 / np.sum(weights ** 2)


@njit
def particle_filter_single_seed(
    measurements,
    n_particles,
    process_noise,
    measurement_noise,
    initial_state,
    seed
):
    """单次确定性粒子滤波"""
    np.random.seed(seed)  # ← 关键：入口播种

    T = len(measurements)
    particles = initial_state + np.random.randn(n_particles) * 2.0
    weights = np.ones(n_particles) / n_particles
    estimates = np.empty(T)

    for t in range(T):
        # 1. 预测
        particles += np.random.randn(n_particles) * process_noise

        # 2. 更新权重
        z_t = measurements[t]
        for i in range(n_particles):
            residual = z_t - particles[i]
            likelihood = np.exp(-0.5 * (residual / measurement_noise) ** 2)
            weights[i] *= likelihood

        # 3. 归一化
        weights_sum = np.sum(weights)
        if weights_sum > 1e-15:
            weights /= weights_sum
        else:
            weights = np.ones(n_particles) / n_particles

        # 4. 估计
        estimates[t] = np.sum(particles * weights)

        # 5. 重采样
        N_eff = effective_sample_size(weights)
        if N_eff < n_particles / 2.0:
            indices = systematic_resample(weights)
            particles = particles[indices]
            weights = np.ones(n_particles) / n_particles

    return estimates


@njit
def multi_seed_particle_filter(
    measurements,
    n_particles,
    process_noise,
    measurement_noise,
    initial_state,
    base_seed,
    n_seeds
):
    """多 Seed 似然加权粒子滤波"""
    T = len(measurements)
    all_estimates = np.empty((n_seeds, T))
    all_likelihoods = np.empty(n_seeds)

    for k in range(n_seeds):
        seed_k = base_seed + k * 1000

        estimates_k = particle_filter_single_seed(
            measurements, n_particles, process_noise,
            measurement_noise, initial_state, seed_k
        )

        all_estimates[k] = estimates_k

        # 计算似然（置信度）
        log_likelihood = 0.0
        for t in range(T):
            residual = measurements[t] - estimates_k[t]
            log_likelihood += -0.5 * (residual / measurement_noise) ** 2

        all_likelihoods[k] = np.exp(log_likelihood)

    # 似然归一化为权重
    likelihood_sum = np.sum(all_likelihoods)
    if likelihood_sum > 1e-15:
        seed_weights = all_likelihoods / likelihood_sum
    else:
        seed_weights = np.ones(n_seeds) / n_seeds

    # 加权平均
    final_estimates = np.empty(T)
    final_std = np.empty(T)

    for t in range(T):
        final_estimates[t] = np.sum(seed_weights * all_estimates[:, t])
        variance = np.sum(seed_weights * (all_estimates[:, t] - final_estimates[t]) ** 2)
        final_std[t] = np.sqrt(variance)

    return final_estimates, final_std, all_estimates, seed_weights


def demo_particle_filter():
    print(f"\n{'='*55}")
    print("案例 8: 多 Seed 似然加权粒子滤波")
    print(f"{'='*55}")

    # 生成模拟数据
    np.random.seed(42)
    T = 100
    process_noise = 0.5
    measurement_noise = 1.0

    # 真实状态：随机游走
    true_states = np.zeros(T)
    true_states[0] = 0.0
    for t in range(1, T):
        true_states[t] = true_states[t-1] + np.random.randn() * process_noise

    # 观测
    measurements = true_states + np.random.randn(T) * measurement_noise

    # 热身
    multi_seed_particle_filter(
        measurements[:10], 100, process_noise, measurement_noise, 0.0, 42, 3
    )

    # 运行多 Seed 粒子滤波
    t0 = time.perf_counter()
    final_est, final_std, all_est, seed_weights = multi_seed_particle_filter(
        measurements=measurements,
        n_particles=500,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_state=0.0,
        base_seed=42,
        n_seeds=10
    )
    t = time.perf_counter() - t0

    # 评估
    rmse = np.sqrt(np.mean((final_est - true_states) ** 2))

    print(f"  时间步数:      {T}")
    print(f"  粒子数:        500")
    print(f"  运行次数:      10 seeds")
    print(f"  计算时间:      {t:.4f}s")
    print(f"  RMSE:          {rmse:.4f}")
    print(f"  平均不确定性:  {np.mean(final_std):.4f}")
    print()
    print("  各 Seed 权重 (似然归一化):")
    for k, w in enumerate(seed_weights):
        print(f"    Seed {k}: {w:.4f}")

    print()
    print("  💡 关键特性:")
    print("     - 100% 可复现（相同base_seed → 相同结果）")
    print("     - 似然加权（好的估计贡献更大）")
    print("     - 不确定性量化（标准差）")

    # 可复现性验证
    final_est2, _, _, _ = multi_seed_particle_filter(
        measurements, 500, process_noise, measurement_noise, 0.0, 42, 10
    )
    print(f"\n  可复现性验证: {np.allclose(final_est, final_est2)}")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════╗")
    print("║    Numba 第十二章：实战案例               ║")
    print("║    机器学习 + Monte Carlo + 数值计算     ║")
    print("╚═══════════════════════════════════════════╝")

    demo_distance_matrix()
    demo_rolling_features()
    demo_kmeans()
    demo_option_pricing()
    demo_pi_estimation()
    demo_gradient_descent()
    demo_numerical_integration()
    demo_particle_filter()

    print(f"\n{'='*55}")
    print("🎉 恭喜完成 Numba 全部 12 章学习！")
    print(f"{'='*55}")
    print()
    print("  下一步:")
    print("  • 把 Numba 用到你的实际项目中")
    print("  • 遇到问题查官方文档 + inspect_types()")
    print("  • 分享你的加速经验")
