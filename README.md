
# My-DINOv2-Retrieval：基于 NumPy 的图像检索系统

本仓库是一个纯 **NumPy** 实现的 DINOv2 (Vision Transformer) 图像检索系统。项目脱离了 PyTorch 等深度学习框架的推理依赖，完整实现了从底层矩阵运算到上层应用检索的全过程，重点攻克了位置编码动态插值与图像维度对齐等核心难题。

---

## 🌟 项目亮点

* **纯 NumPy 推理**：手动实现 Transformer Block、多头自注意力机制（Multi-Head Attention）及 LayerNorm 等底层算子，而非调用深度学习库。
* **动态尺寸适配**：针对 ViT 固定的位置编码，实现了基于 `scipy.ndimage.zoom` 的动态插值函数，支持处理任意长宽比的 Patch 网格。
* **工程化设计**：包含多线程数据爬取、模型精度验证（MAE 误差控制在 $10^{-6}$）、特征库归一化构建及 Top-K 余弦相似度检索。

---

## 📂 目录结构

```text
My-DINOv2-Retrieval/
├── demo_data/                # 验证数据集
│   ├── cat.jpg               # 测试图片（猫）
│   ├── dog.jpg               # 测试图片（狗）
│   └── cat_dog_feature.npy   # 官方标准参考特征（用于精度比对）
├── gallery_images/           # 图库文件夹（下载后的图片存放于此）
├── dinov2_numpy.py           # 核心模型代码（Transformer 架构与插值实现）
├── preprocess_image.py       # 图像预处理（包含 14-patch 强制对齐逻辑）
├── debug.py                  # 精度验证脚本（计算与标准特征的误差）
├── download_data.py          # 多线程下载脚本（从 CSV 自动化获取图片）
├── image_retrieval.py        # 检索系统主程序（建库、归一化、Top-K 搜索）
├── data.csv                  # 包含万级图片 URL 的数据源
├── requirements.txt          # 项目依赖列表
└── README.md                 # 本项目说明文档

```

---

## 🧠 核心技术逻辑

### 1. 动态尺寸对齐 (preprocess_image.py)

由于 DINOv2 要求输入必须能被 Patch Size (14) 整除，且为了保持长宽比，我们实现了 `resize_short_side` 逻辑：

* **逻辑**：找出短边，将其缩放到 224，长边按比例缩放。
* **关键点**：最终尺寸通过 `round(L / 14) * 14` 强制对齐，确保特征提取时不会因维度不匹配报错。

### 2. 位置编码插值 (dinov2_numpy.py)

预训练权重（Base 版）固定对应  个 Patch。当输入分辨率改变时，通过以下流程适配：

* 将原始 1D 位置编码还原为 2D 空间网格。
* 使用双线性插值将其“拉伸”或“压缩”至当前输入所需的 Patch 数量。
* 重新拼接 CLS Token，完成位置感知。

### 3. 多头注意力机制 (Attention)

基于矩阵乘法模拟了注意力分数的计算过程：



利用 `reshape` 和 `transpose` 操作，将 768 维特征拆分为 12 个独立的 Head 并行计算。

---

## 🚀 运行指南

### 第一步：环境安装

```bash
pip install -r requirements.txt

```

### 第二步：权重准备

确保项目根目录下有 `vit-dinov2-base.npz` 文件。

### 第三步：下载图库

利用多线程爬虫脚本批量获取检索图库：

```bash
python download_data.py

```

### 第四步：模型精度验证 (Debug)

对比 NumPy 模型输出与标准答案，确保 MAE 误差在允许范围内：

```bash
python debug.py

```

### 第五步：启动以图搜图

1. **建库**：程序会自动提取 `gallery_images` 路径下所有图库图片的特征。
2. **搜索**：输入查询图片路径，系统将基于 L2 归一化后的特征进行余弦相似度计算，返回最接近的结果。

```bash
python image_retrieval.py

```

---

## 📈 实验表现

* **数值稳定性**：在  输入下，模型输出特征与官方特征的 MAE 误差极小，证明了逻辑的正确性。
* **检索速度**：得益于矩阵化运算，在 1000+ 张图片规模下，相似度匹配速度可达毫秒级。

---

## 👤 信息


* **仓库**：[GitHub/YourUsername/My-DINOv2-Retrieval]

```
