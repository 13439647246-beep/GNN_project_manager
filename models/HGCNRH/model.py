# ============================================================
# models/HGCNRH/model.py
# HGCNRH：模块度 + 高阶节点信息 + 双仿射注意力 + 变分重构
#
# 适配说明：
#   - 真实特征数据集（Cora / CiteSeer / Amazon / Wikipedia 等）：
#       in_dim 较大，用 LayerNorm(in_dim) 归一化；
#   - 单特征数据集（PolBlogs 等）：
#       LayerNorm(1) 会把每行归零成常数，改用 BatchNorm1d(1)；
#   - 无特征数据集：loader 层负责构造度特征，模型要求 num_features >= 1。
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from src.utils.registry import register_model
from models.HGCNRH.layer import HGCNRHLayer
from src.ops.linalg import modularity_score, modularity_matrix


@register_model("HGCNRH")
def build_model(arg):
    """注册的构建函数：arg → 模型实例"""
    return HGCNRH(arg)


# ------------------------------------------------------------
# 图上的注意力融合：用模块度矩阵 B_o 作 query、A_fused 作 key
# 输出是 data.x 的加权组合，形状与 data.x 相同
# ------------------------------------------------------------
def attention(q, key, v, mask=None):
    """
    Q: (n, n)
    K: (n, n)
    V: (n, δ)
    返回:
        out:  (n, δ)
        attn: (n, n)
    """
    d_k = q.size(-1)  # = n
    scores = q @ key.transpose(-2, -1) / (d_k ** 0.5)  # (n, n)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attn_p = F.softmax(scores, dim=-1)  # (n, n)
    out = attn_p @ v                    # (n, δ)
    return out, attn_p


def build_augmented_x(B_o, A_fused, data):
    """结构引导的特征增强：out 与 data.x 同形状"""
    out, attn = attention(B_o, A_fused, data.x)
    return out, attn


class HGCNRH(nn.Module):
    def __init__(self, arg):
        super().__init__()
        cfg = arg["model"]["kwargs"]

        # ---------- 基本维度 ----------
        in_dim = int(arg["data"]["num_features"])
        hidden_dim = cfg.get("hidden_dim", 128)
        qk_dim = cfg.get("qk_dim", 64)
        gcn_layers = cfg.get("gcn_layers", 1)
        dropout = cfg.get("dropout", 0.05)

        # ---------- 社区数与 KMeans 配置 ----------
        self.K = int(arg["data"]["num_classes"])
        # 兼容旧名 kmeans_n，新名 kmeans_n_init 更语义化
        self.kmeans_n_init = cfg.get("kmeans_n_init", cfg.get("kmeans_n", 10))

        # ---------- 特征归一化 ----------
        # 1 维特征时 LayerNorm 会把每行输出恒等于 beta（常数），
        # 节点之间差异全部丢失；改用 BatchNorm1d 跨节点归一化，保留相对差异。
        if in_dim == 1:
            self.norm_x = nn.BatchNorm1d(in_dim)
        elif in_dim >= 2:
            self.norm_x = nn.LayerNorm(in_dim)
        else:
            raise ValueError(
                f"HGCNRH 要求 num_features >= 1，当前 data.num_features={in_dim}。"
                f"无特征数据集请在 loader 中先构造特征。"
            )

        # ---------- 主体层 ----------
        self.layer = HGCNRHLayer(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            qk_dim=qk_dim,
            out_dim=self.K,
            gcn_layers=gcn_layers,
            dropout=dropout,
        )

        # ---------- 预计算缓存（不可学习） ----------
        self.register_buffer("_cached_x", torch.empty(0), persistent=False)
        self.register_buffer("_B_o", torch.empty(0), persistent=False)
        self.register_buffer("_A_fused", torch.empty(0), persistent=False)

    # ========================================================
    # 预处理：缓存归一化后的增强特征、模块度矩阵 B_o、融合邻接 A_fused
    # ========================================================
    def prepare_input(self, data):
        # 确保模型与数据同设备
        if next(self.parameters()).device != data.x.device:
            self.to(data.x.device)

        self._prepare_B_o(data)
        self._prepare_A_fused(data)

        if self._cached_x.numel() == 0 or self._cached_x.device != data.x.device:
            with torch.no_grad():
                aug, _ = build_augmented_x(self._B_o, self._A_fused, data)
                x_fused = data.x + aug
                # BatchNorm 在 eval 模式下用 running stats；
                # 此处 no_grad，BN 仍会更新 running stats（无梯度也可更新）
                # self._cached_x = self.norm_x(x_fused)
                self._cached_x = self.norm_x(x_fused)

    def _prepare_B_o(self, data):
        """模块度矩阵 B = A - k k^T / 2m，稠密 (n, n)"""
        if self._B_o.numel() == 0 or self._B_o.device != data.x.device:
            with torch.no_grad():
                A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                k = A.sum(dim=1)
                m = A.sum() / 2
                self._B_o = A - torch.outer(k, k) / (2 * m).clamp(min=1e-12)
        return self._B_o

    def _prepare_A_fused(self, data):
        """A_fused = A + A^2 ⊙ A，稀疏 (n, n)，用于注入高阶邻居信息"""
        if self._A_fused.numel() == 0 or self._A_fused.device != data.x.device:
            with torch.no_grad():
                n = data.num_nodes
                ei = data.edge_index

                # 1) 无向二值邻接（去自环、去重）
                row = torch.cat([ei[0], ei[1]])
                col = torch.cat([ei[1], ei[0]])
                keep = row != col
                row, col = row[keep], col[keep]

                flat = torch.unique(row * n + col)
                row, col = flat // n, flat % n

                vals = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
                A = torch.sparse_coo_tensor(
                    torch.stack([row, col]), vals, size=(n, n)
                ).coalesce()

                # 2) 稀疏 A^2
                A2 = torch.sparse.mm(A, A)

                # 3) A^2 ⊙ A：只保留 A 支持集上的项
                a_idx = A._indices()
                a_key = a_idx[0] * n + a_idx[1]

                a2_idx = A2._indices()
                a2_key = a2_idx[0] * n + a2_idx[1]

                keep2 = torch.isin(a2_key, a_key)
                A2_on_A = torch.sparse_coo_tensor(
                    a2_idx[:, keep2], A2._values()[keep2], size=(n, n),
                ).coalesce()

                self._A_fused = (A + A2_on_A).coalesce()

        return self._A_fused

    # ========================================================
    # 前向
    # ========================================================
    def forward(self, data):
        if self._cached_x.numel() == 0 or self._cached_x.device != data.x.device:
            self.prepare_input(data)

        x = self._cached_x
        B_o = self._B_o

        C, mu, beta = self.layer(x, data.edge_index)
        return {"C": C, "mu": mu, "beta": beta, "B_o": B_o}

    # ========================================================
    # 评估：KMeans on C，多次采样取平均
    # ========================================================
    @torch.no_grad()
    def evaluate(self, data, split="test", n_samples=None):
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score
        import numpy as np

        self.eval()
        true_labels = data.y.cpu().numpy()

        # C 带采样噪声，val/test 多次采样取平均；train 只跑 1 次
        if n_samples is None:
            n_samples = 1 if split == "train" else 5
        n_init = 1 if split == "train" else self.kmeans_n_init

        qps, nmis = [], []
        for _ in range(n_samples):
            outputs = self(data)
            feat = outputs["C"].detach().cpu().numpy()

            kmeans = KMeans(n_clusters=self.K, n_init=n_init, random_state=42)
            pred_labels = kmeans.fit_predict(feat)

            qps.append(modularity_score(data.edge_index, pred_labels, num_nodes=data.num_nodes))
            nmis.append(normalized_mutual_info_score(true_labels, pred_labels))

        return {"Qp": float(np.mean(qps)), "NMI": float(np.mean(nmis))}