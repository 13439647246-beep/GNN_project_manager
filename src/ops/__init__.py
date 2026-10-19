# ============================================================
# src/ops/__init__.py
# 数理运算库对外接口。
# 外部只需 from src.ops import accuracy, cross_entropy, ...
# ============================================================

# ---- 概率运算 / 掩码 ----
from src.ops.functional import (
    softmax, log_softmax, masked_softmax,
    masked_select, masked_mean, masked_sum,
)

# ---- 分类损失 ----
from src.ops.functional import (
    cross_entropy, nll_loss,
    binary_cross_entropy, binary_cross_entropy_with_logits,
)

# ---- 回归损失 ----
from src.ops.functional import (
    mse_loss, l1_loss, smooth_l1_loss,
)

# ---- 分布距离 ----
from src.ops.functional import kl_divergence

# ---- 对比 / 度量学习损失 ----
from src.ops.functional import info_nce_loss, triplet_loss

# ---- 图相关 ----
from src.ops.functional import scatter_softmax

# ---- 线代 ----
from src.ops.linalg import (
    l2_normalize,
    cosine_similarity, pairwise_cosine_similarity,
    pairwise_euclidean_distances,
    top_k_indices,
    symmetric_normalize_adj, add_self_loops_dense, modularity_score,modularity_matrix,
)

# ---- 指标 ----
from src.ops.metrics import (
    accuracy, top_k_accuracy,
    confusion_matrix, precision_recall_f1,
    mse, mae, r2_score,
    auc_roc,
)

# ---- 杂项工具 ----
from src.ops.utils import (
    to_tensor, to_numpy, to_scalar,
    to_one_hot, from_one_hot,
    count_parameters, print_model_summary,
    safe_divide, log1p_exp,
    flatten_dict, merge_dicts,
)

__all__ = [
    # functional
    "softmax", "log_softmax", "masked_softmax",
    "masked_select", "masked_mean", "masked_sum",
    "cross_entropy", "nll_loss",
    "binary_cross_entropy", "binary_cross_entropy_with_logits",
    "mse_loss", "l1_loss", "smooth_l1_loss",
    "kl_divergence",
    "info_nce_loss", "triplet_loss",
    "scatter_softmax",
    # linalg
    "l2_normalize",
    "cosine_similarity", "pairwise_cosine_similarity",
    "pairwise_euclidean_distances",
    "top_k_indices",
    "symmetric_normalize_adj", "add_self_loops_dense", "modularity_score","modularity_matrix",
    # metrics
    "accuracy", "top_k_accuracy",
    "confusion_matrix", "precision_recall_f1",
    "mse", "mae", "r2_score",
    "auc_roc",
    # utils
    "to_tensor", "to_numpy", "to_scalar",
    "to_one_hot", "from_one_hot",
    "count_parameters", "print_model_summary",
    "safe_divide", "log1p_exp",
    "flatten_dict", "merge_dicts",
]
