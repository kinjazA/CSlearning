# 01 图像基本概念

![image-20260212200048728](F:\note\deep_learning\pytorch_learning\day06_图像相关.assets\image-20260212200048728.png)

![image-20260212083759328](F:\note\deep_learning\pytorch_learning\day06_图像相关.assets\image-20260212083759328.png)

- 看“**位深度**”或“**位深度/颜色深度**”这一项：
   - 8位 → 通常是**1通道**（灰度图）
   - 24位 → **3通道**（RGB普通彩色）
   - 32位 → **4通道**（RGBA带透明通道）

这个方法**不完全准确**（有些软件显示的总位深不一定严格对应通道×位深），但对日常JPG/PNG 90%的情况下是够用的快捷判断。

# 02 matplotlib.pyplot图像处理函数

### `plt.imread()` —— 读取图像

> 它的作用是把一张图片文件（如 .jpg, .png）加载到内存中，变成一个 **NumPy 数组**。

```python
img_array = plt.imread(fname, format=None)
```

### `plt.imshow()` —— 显示图像

> 这是最复杂的函数，它的功能极其强大。它不仅仅是“把图画出来”，它还能帮你做**伪彩色映射**、**插值平滑**和**数据归一化**。
>
> * **`X`** (必填): 图像数据。
>
>    - 如果是 **(M, N)** 的二维数组：会被当作**灰度图**或**热力图**处理。
>    - 如果是 **(M, N, 3)** 的三维数组：会被当作 **RGB 彩色图**处理。
>    - 如果是 **(M, N, 4)** 的三维数组：会被当作 **RGBA（带透明度）** 处理。
>
>    **`cmap`** (Colormap, 颜色映射): **仅对二维数组（灰度图）有效**。
>
>    - 如果不填，Matplotlib 默认使用 `'viridis'`（一种黄绿紫的颜色），这会导致你的黑白照片看起来像热成像。
>    - **常用值**：`'gray'` (黑白), `'hot'` (热力图), `'jet'` (彩虹色), `'binary'` (二值化)。
>    - - `vmax`** (值域控制): **非常重要！**
>    - 用于控制颜色的显示范围。
>    - 例如，你的图像数据里有噪音，范围是 0~1000，但你想突出 100~200 之间的细节。
>    - *写法示例*：`plt.imshow(data, vmin=100, vmax=200)` (小于100显示为最黑，大于200显示为最亮)。
>
>    **`interpolation`** (插值方式):
>
>    - 当图片像素少、屏幕大时，怎么填充中间的空隙？
>    - `'nearest'`：最近邻插值。你会看到明显的马赛克方块（适合看像素细节）。
>    - `'bicubic'`：双三次插值。会让图片变模糊、平滑（适合看自然照片）。
>    - *默认值*：通常是 `antialiased`（抗锯齿）。

```python
plt.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, ...)
```

### `plt.imsave()` —— 保存图像数组

它的作用是将一个 **NumPy 数组** 保存为图片文件。

> **注意**：它和 `plt.savefig()` 不一样。`savefig` 是保存整个画板（包括坐标轴、标题、白边）；而 `imsave` 只保存图像本身的内容，不带坐标轴。

```python
plt.imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, ...)
```

```python
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 模拟数据 
# ==========================================
# 创建一个 100x100 的随机彩色图 (RGB)
img_rgb = np.random.rand(100, 100, 3) 

# 创建一个 100x100 的渐变图 (灰度)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
img_gray = np.sin(X**2 + Y**2)

# ==========================================
# 2. 实战 imshow (显示)
# ==========================================
plt.figure(figsize=(10, 5))

# --- 子图 1: 显示彩色图 ---
plt.subplot(1, 3, 1)
plt.title("RGB Random Noise")
# interpolation='nearest' 能看清每一个像素格
plt.imshow(img_rgb, interpolation='nearest') 
plt.axis('off') #以此关闭坐标轴

# --- 子图 2: 显示灰度图 (错误示范) ---
plt.subplot(1, 3, 2)
plt.title("Gray (Default cmap)")
# 如果不指定 cmap='gray'，它默认是绿紫色的(viridis)
plt.imshow(img_gray) 
plt.axis('off')

# --- 子图 3: 显示灰度图 (正确示范) ---
plt.subplot(1, 3, 3)
plt.title("Gray (cmap='gray')")
# 指定 cmap='gray' 才是真正的黑白照
plt.imshow(img_gray, cmap='gray') 
plt.axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# 3. 实战 imsave (保存)
# ==========================================
print("正在保存图片...")

# 保存那张灰度图，但是在保存时给它上色(变成热力图风格)
# 这在深度学习可视化 Feature Map 时非常有用
plt.imsave('my_heatmap.png', img_gray, cmap='hot')

print("保存成功: my_heatmap.png")

# ==========================================
# 4. 实战 imread (读取)
# ==========================================
# 读取刚才保存的文件
img_read = plt.imread('my_heatmap.png')

print(f"读取后的形状: {img_read.shape}")
print(f"读取后的数据类型: {img_read.dtype}")
# 注意：imsave 保存时通常会包含 Alpha 通道(透明度)，所以形状可能是 (100, 100, 4)
```
