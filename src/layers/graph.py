# ============================================================
# src/layers/graph.py
# 图相关块：在 PyG 卷积层之上做"Conv + Norm + Act + Dropout"的组合封装。
#
# 定位：
#   - PyG 提供"卷积"原子层（GCNConv / GATConv / SAGEConv / GINConv …）
#   - 本文件提供"块"级封装，方便搭模型时一行组合
#
# 约定：本文件里所有 block 的卷积接口统一为 conv(x, edge_index)，
#       这是 GCN / GAT / SAGE 等主流卷积的共同签名。
# ============================================================

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv,
)

from src.layers.basic import get_activation


# ============================================================
# 通用图卷积块
# ============================================================
class GCN(nn.Module):
    """
    通用图卷积块：Conv → Norm → Act → Dropout，可选残差。

    参数:
        conv        : 已经实例化的 PyG 卷积层（如 GCNConv(in, out)）
        in_dim      : 输入维度（残差判断用）
        out_dim     : 输出维度（残差判断用）
        norm        : None / "batch" / "layer"
        activation  : 激活名（None 表示不加）
        dropout     : dropout 概率
        residual    : 是否使用残差连接
    """
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers = 1,
                 norm=None,
                 activation="relu",
                 dropout=0.0,
                 residual=False):
        super().__init__()
        # 归一化（可选）
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f"未知 norm: {norm}，可选 None / 'batch' / 'layer'")

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GCNConv(in_dim, out_dim))
        else:
            self.layers.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(GCNConv(hidden_dim, out_dim))

        self.activation     = get_activation(activation)
        self.dropout = dropout

        # 残差：维度不一致时用线性投影对齐
        self.residual = residual
        if residual:
            self.res_proj = (nn.Identity() if in_dim == out_dim
                             else nn.Linear(in_dim, out_dim, bias=False))

    def forward(self, out, edge_index):
        for i, conv in enumerate(self.layers):
            out = conv(out, edge_index)
            # 最后一层不加激活和 dropout
            if i != len(self.layers) - 1:
                out = F.tanh(out)
                out = F.dropout(out, p=self.dropout, training=self.training)

        # 残差分支
        if self.residual:
            out = out + self.res_proj(out)
        return out