# ============================================================
# src/layers/__init__.py
# 通用层对外接口。
# 外部只需 from src.layers import MLP, GNNBlock, ...
# ============================================================

# 基础层
from src.layers.basic import (
    get_activation,
    MLP,
    ResidualBlock,
)

# 图相关块
from src.layers.graph import (
    GCN,
)

# 注意力
from src.layers.attention import MultiHeadSelfAttention, BiaffineAttention

# 初始化
from src.layers.initialization import (
    apply_init,
    xavier_init,
    kaiming_init,
    orthogonal_init,
    normal_init,
)

__all__ = [
    # basic
    "get_activation", "MLP", "ResidualBlock",
    # graph
    "GCN",
    # attention
    "MultiHeadSelfAttention", "BiaffineAttention",
    # initialization
    "apply_init",
    "xavier_init", "kaiming_init", "orthogonal_init", "normal_init",
]