# H&M Personalized Fashion Recommendations

> **课程**：大数据分析与计算 (610ZH125)
> **团队**：第 5 组 · 马峥、祝帅
> **Kaggle 排名**：116 / 2952（Top 4%）

---

## 项目概述

基于 H&M 近两年 3178 万条真实交易数据，构建个性化时尚推荐系统。核心方案采用 **4 路召回 + PyTorch GPU SVD 嵌入 + LightGBM/CatBoost 双模型融合排序** 的工业级"召回 + 排序"架构。

- **任务**：预测用户未来 7 天最可能购买的 12 件商品
- **评估指标**：MAP@12
- **数据规模**：3178 万交易记录 / 137 万用户 / 10.5 万商品
- **时间跨度**：2018.09 — 2020.09（两年完整季节周期）

---

## 目录结构

```
hm-github/
├── README.md                        # 本文件
├── requirements.txt                 # Python 依赖
├── .gitignore                       # Git 忽略规则
│
├── notebooks/                       # 源代码（Jupyter Notebook）
│   ├── 01_data_preprocessing.ipynb  # 数据加载与预处理
│   ├── 02_feature_engineering.ipynb # 特征工程（LightFM + 聚合特征）
│   ├── 03_train_infer.ipynb         # 训练 + 推理 + Submission
│   └── 04_gpu_accelerated.ipynb     # GPU 加速版（PyTorch SVD 替代 LightFM）
│
└── reports/                         # 项目报告
    ├── 小组大作业报告.md              # 小组大作业报告
    └── 个人大作业报告_马峥.md          # 马峥个人大作业报告
```

---

## 技术架构

### 整体流水线

```
原始数据 (CSV)
    │
    ├── CSV → Pickle 转换
    │
    ▼
    ┌──────────────────────────────────────┐
    │          数据预处理与 ID 映射          │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │        4 路召回（每用户 20 个候选）     │
    │  ① 复购召回 ── 最近购买的优先推荐       │
    │  ② Item2Item ── 相似商品共现推荐       │
    │  ③ 全局热门 ── 全平台爆款兜底          │
    │  ④ 品类热门 ── 用户偏好类目热门补位    │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │     LightFM / PyTorch SVD 嵌入        │
    │     64 维 user/item 向量              │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │             特征工程                   │
    │  • 用户/商品/交互 三层动态特征         │
    │  • 用户 × 属性聚合特征（402 维）       │
    │  • PCA 降维（GPU，402→32 维）         │
    │  • LightFM 嵌入特征                   │
    └──────────────────┬───────────────────┘
                       │
    ┌──────────────────▼───────────────────┐
    │     LightGBM + CatBoost 融合排序      │
    │     3-fold GroupKFold 验证            │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │    生成 submission.csv (MAP@12 格式)  │
    └──────────────────────────────────────┘
```

### 召回策略（4 路）

| 策略 | 权重 | 说明 |
|:---|---:|:---|
| 复购召回 | 40% | 按 `10⁹ × day_rank + volume_rank` 排序 |
| Item2Item | 30% | 基于购物篮共现矩阵的相似商品推荐 |
| 品类热门 | 20% | 用户偏好品类内的热门商品 |
| 全局热门 | 10% | 全平台热门商品兜底（冷启动） |

### 特征体系

- **嵌入特征**：LightFM BPR / PyTorch SVD 64 维 user/item 向量
- **用户动态特征**：price/sales_channel 统计量、购买次数、活跃天数
- **商品动态特征**：price 统计量、被购次数、新鲜度
- **交互对特征**：购买次数、距上次购买时间、平均价格
- **用户 × 属性聚合**：商品 One-hot → 交易聚合 → Groupby user 均值 → PCA 降维

### GPU 加速

| 阶段 | 技术 | GPU |
|:---|:---|---:|
| 矩阵分解 | PyTorch `torch.svd_lowrank` | Tesla P100 16GB |
| 排序模型 | LightGBM `device=gpu` | Tesla P100 |
| 排序模型 | CatBoost `task_type=GPU` | Tesla P100 |
| PCA 降维 | PyTorch GPU 矩阵乘法 | Tesla P100 |

---

## 运行说明

### 环境要求

- Python 3.7+
- GPU（推荐 Tesla P100 16GB 或同等，非必需但会大幅加速）
- 内存 ≥ 32GB（推荐 64GB）
- Kaggle API 账号（用于下载数据）

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载竞赛数据
#    https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
#    将数据放在 data/ 目录下

# 3. 按顺序运行 Notebook
#    notebooks/01_data_preprocessing.ipynb
#    notebooks/02_feature_engineering.ipynb
#    notebooks/03_train_infer.ipynb
```

### 数据下载

```python
import kagglehub
kagglehub.competition_download(
    'h-and-m-personalized-fashion-recommendations',
    output_dir='./data'
)
```

---

## 核心发现

1. **复购是最强预测信号**：复购特征权重最高，近 31 天行为最有效
2. **商品热度是第二强信号**：冷启动用户靠热门兜底最稳定
3. **年龄是"伪重要特征"**：显示重要但移除后性能不变，弃用
4. **时序验证不可替代**：随机 K 折导致分数虚高 2-3 倍
5. **双模型融合优于单模型**：方差降低 40%

---

## Kaggle

- **竞赛地址**：https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
- **最终排名**：116 / 2952（Top 4%）
- **方案亮点**：全流程 GPU 加速（SVD / LightGBM / CatBoost）

---

## 参考资料

- [Kaggle H&M 竞赛页面](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- [LightFM — BPR Bayesian Personalized Ranking](https://making.lyst.com/lightfm/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
