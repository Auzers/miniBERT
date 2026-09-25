# miniBERT

本项目是斯坦福 CS224N（Spring 2024）默认最终项目，实现了一个简化版 BERT（minBERT），并将其用于三个下游任务：

- **情感分类（Sentiment）**：SST 数据集，5 分类
- **复述检测（Paraphrase）**：Quora 数据集，二分类
- **语义相似度（STS）**：SemEval 数据集，回归（0~5）

## 项目结构

项目分两部分：

### Part 1：实现 BERT 核心组件

| 文件 | 内容 |
|---|---|
| `bert.py` | 多头注意力、Add & Norm、BERT 层前向、embedding |
| `optimizer.py` | AdamW 优化器（含解耦权重衰减 + 偏差修正） |
| `classifier.py` | 单任务情感分类（SST / CFIMDB） |

### Part 2：多任务 BERT

| 文件 | 说明 |
|---|---|
| `multitask_classifier.py` | 个性化修改使模型能同时用于上述三个下游任务|

## 扩展与提升

### 1. 丰富特征

STS 任务最初用 `[u; v]` 拼接两个句子向量。改进为：

```
[u; v; |u − v|; u · v]
```

显式加入"逐元素差"（`|u − v|`）和"逐元素乘积"（`u · v`），把相似度的直接线索提供给模型，而不是让它从原始拼接里自己学习。

### 2. 对比学习（Multiple Negatives Ranking Loss）

从 Quora 中筛出正例对（`is_duplicate == 1`），用对比学习优化 embedding：让相似句对的向量靠拢、不相似的分开。训练时以 batch 内其他句子对为负例，对每个句子对算余弦相似度矩阵，用 softmax 交叉熵逼模型给"正确配对"打最高分。

## 实验结果

### Part 1：单任务情感分类（Dev Accuracy）

| 配置 | Dev Accuracy |
|---|---|
| SST + last-linear-layer | 0.385 |
| CFIMDB + last-linear-layer | 0.739 |
| SST + full-model | 0.524 |
| CFIMDB + full-model | 0.967 |

### Part 2：多任务（Dev 指标）

| 版本 | Sentiment acc | Paraphrase acc | STS corr |
|---|---|---|---|
| 基线（`[u; v]` 拼接） | 0.525 | 0.735 | 0.352 |
| 丰富特征（`[u; v; \|u−v\|; u·v]`） | 0.517 | 0.746 | 0.462 |
| **+ 对比学习（MNRL）** | 0.500 | 0.753 | **0.540** |

经过两轮扩展，STS 相关系数从 **0.352 → 0.540**（+0.188）：丰富特征 +0.11，对比学习再 +0.08。paraphrase 稳步提升（0.735 → 0.753）。sentiment 略有回落（0.525 → 0.500），是对比学习扰动共享 BERT 的正常代价；综合分从 0.537 提升到 0.598。

