# src/ops

无状态的数理运算与损失函数。

## 文件职责

| 文件 | 内容 |
|---|---|
| `functional.py` | 概率运算、掩码运算、通用损失（分类/回归/对比/图） |
| `linalg.py` | 相似度、距离、Top-K、邻接矩阵归一化 |
| `metrics.py` | 分类/回归指标：acc、F1、AUC、MSE、R² |
| `utils.py` | 类型转换、one-hot、参数统计、数值稳定工具 |

## 与 `models/<name>/loss.py` 的边界

- **通用损失**（交叉熵、MSE、InfoNCE…）→ `ops/functional.py`
- **模型特定损失**（带正则、特定加权、图掩码组合）→ `models/<name>/loss.py`
- 模型损失里可以调用 `ops` 里的通用函数，不重复造轮子

## 使用示例

```python
from src.ops import (
    accuracy, precision_recall_f1, cross_entropy,
    masked_mean, pairwise_cosine_similarity,
)

# 指标
acc = accuracy(logits, labels)
metrics = precision_recall_f1(logits, labels, average="macro")

# 掩码均值（图模型里的节点筛选）
loss = cross_entropy(logits[train_mask], labels[train_mask])