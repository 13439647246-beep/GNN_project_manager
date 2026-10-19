import math

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


# 注意力融合信息
def attention(q, key, v, mask=None):
    """
    Q: (n, n)
    K: (n, n)
    V: (n, k)
    返回:
        out:  (n, k)
        attn: (n, n)
    """
    d_k = q.size(-1)  # = n
    scores = q @ key.transpose(-2, -1) / (d_k ** 0.5)  # (n, n)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attn_p = F.softmax(scores, dim=-1)  # (n, n)
    out = attn_p @ v  # (n, k)

    return out, attn_p


def build_augmented_x(B_o, A_fused, data):
    out, attn = attention(B_o, A_fused, data.x)
    return out, attn


class HGCNRH(nn.Module):
    def __init__(self, arg):
        super().__init__()
        cfg = arg["model"]["kwargs"]

        in_dim = arg["data"]["num_features"]
        hidden_dim = cfg.get("hidden_dim", 128)
        qk_dim = cfg.get("qk_dim", 64)
        gcn_layers = cfg.get("gcn_layers", 1)
        dropout = cfg.get("dropout", 0.05)

        self.K = arg["data"]["num_classes"]
        self.kmeans_n = cfg.get("kmeans_n", 10)
        self.hidden_dim = hidden_dim

        self.norm_x = nn.LayerNorm(in_dim)
        self.layer = HGCNRHLayer(in_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 qk_dim=qk_dim,
                                 out_dim=self.K,
                                 gcn_layers=gcn_layers,
                                 dropout=dropout)

        self.register_buffer("_A_norm", torch.empty(0), persistent=False)
        self.register_buffer("_B_o", torch.empty(0), persistent=False)
        self.register_buffer("_A_fused", torch.empty(0), persistent=False)

        # __init__ 新增
        self.f_proj = nn.Linear(in_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fuse_norm = nn.LayerNorm(hidden_dim)

        # self._A_norm 用 register_buffer 或普通属性

    def _fuse(self, x, A_norm, B_o):
        F_p = self.f_proj(x)  # (n, d)
        H_A = torch.sparse.mm(A_norm, F_p)  # 三角形结构聚合
        B_n = B_o / (B_o.abs().sum(1, keepdim=True) + 1e-12)
        H_B = B_n @ F_p  # 模块度结构聚合
        H_s = H_A + H_B
        g = torch.sigmoid(self.gate(torch.cat([F_p, H_s], dim=-1)))
        H = self.fuse_norm(g * H_s + (1 - g) * F_p)
        return H  # (n, d)

    def prepare_input(self, data):
        if next(self.parameters()).device != data.x.device:
            self.to(data.x.device)
        self._prepare_B_o(data)
        self._prepare_A_norm(data)  # 新增：归一化后的 A_fused

    def _prepare_B_o(self, data):
        if self._B_o.numel() == 0 or self._B_o.device != data.x.device:
            with torch.no_grad():
                A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                k = A.sum(dim=1)
                m = A.sum() / 2
                self._B_o = A - torch.outer(k, k) / (2 * m).clamp(min=1e-12)
        return self._B_o

    def _prepare_A_norm(self, data):
        device = data.x.device
        if (self._A_norm is not None
                and self._A_norm.numel() > 0
                and self._A_norm.device == device):
            return self._A_norm
        A = self._prepare_A_fused(data)
        n = data.num_nodes
        deg = torch.sparse.sum(A, dim=1).to_dense()
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D = torch.sparse_coo_tensor(
            torch.stack([torch.arange(n, device=A.device)] * 2),
            d_inv_sqrt, size=(n, n)).coalesce()
        self._A_norm = torch.sparse.mm(torch.sparse.mm(D, A), D).coalesce()

    def _prepare_A_fused(self, data):
        if self._A_fused.numel() == 0 or self._A_fused.device != data.x.device:
            print(f"[dbg] 重算 A_fused（epoch 里不该看到这行）")
            with torch.no_grad():
                n = data.num_nodes
                ei = data.edge_index

                # ---- 1. 构造无向二值邻接（稀疏、float32）----
                row = torch.cat([ei[0], ei[1]])
                col = torch.cat([ei[1], ei[0]])
                keep = row != col  # 去自环
                row, col = row[keep], col[keep]

                # 去重（同一条边可能在 ei 里出现两次）
                flat = torch.unique(row * n + col)
                row, col = flat // n, flat % n

                vals = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
                A = torch.sparse_coo_tensor(
                    torch.stack([row, col]), vals, size=(n, n)
                ).coalesce()

                # ---- 2. 稀疏 A² ----
                A2 = torch.sparse.mm(A, A)

                # ---- 3. A2 ⊙ A：只保留 A 支持集上的项 ----
                a_idx = A._indices()  # (2, E_A)
                a_key = a_idx[0] * n + a_idx[1]  # (E_A,)

                a2_idx = A2._indices()
                a2_key = a2_idx[0] * n + a2_idx[1]

                keep2 = torch.isin(a2_key, a_key)
                A2_on_A = torch.sparse_coo_tensor(
                    a2_idx[:, keep2], A2._values()[keep2], size=(n, n),
                ).coalesce()

                # ---- 4. 融合 = A + A²⊙A（保留稀疏）----
                self._A_fused = (A + A2_on_A).coalesce()

        return self._A_fused

    def forward(self, data):
        if (self._B_o.numel() == 0
                or self._B_o.device != data.x.device
                or self._A_norm is None
                or self._A_norm.device != data.x.device):
            self.prepare_input(data)
        B_o = self._B_o
        A_norm = self._A_norm
        x_fused = self._fuse(data.x, A_norm, B_o)  # ★ 每个方案实现不同
        C, mu, beta = self.layer(x_fused, data.edge_index)
        return {"C": C, "mu": mu, "beta": beta, "B_o": B_o}

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
        n_init = 1 if split == "train" else self.kmeans_n

        qps, nmis = [], []
        for _ in range(n_samples):
            outputs = self(data)
            feat = outputs["C"].detach().cpu().numpy()  # ★ 用 C

            kmeans = KMeans(n_clusters=self.K, n_init=n_init, random_state=42)
            pred_labels = kmeans.fit_predict(feat)

            qps.append(modularity_score(data.edge_index, pred_labels, num_nodes=data.num_nodes))
            nmis.append(normalized_mutual_info_score(true_labels, pred_labels))

        return {"Qp": float(np.mean(qps)), "NMI": float(np.mean(nmis))}
