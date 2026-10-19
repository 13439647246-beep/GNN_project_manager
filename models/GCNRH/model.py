import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj

from src.utils.registry import register_model
from models.GCNRH.layer import GCNRHLayer
from src.ops.linalg import modularity_score, modularity_matrix

@register_model("GCNRH")
def build_model(arg):
    """注册的构建函数：arg → 模型实例"""
    return GCNRH(arg)


def build_augmented_x(data):
    """
    模块度与网络特征拼接成新的特征。
    """
    B = modularity_matrix(data.edge_index, num_nodes=data.num_nodes)
    # 拼接
    return torch.cat([B, data.x], dim=-1)


class GCNRH(nn.Module):
    def __init__(self, arg):
        super().__init__()
        cfg = arg["model"]["kwargs"]

        in_dim   = arg["data"]["num_nodes"] + arg["data"]["num_features"]
        out_dim     = arg["data"]["num_classes"]
        hidden_dim  = cfg.get("hidden_dim", 128)
        qk_dim   = cfg.get("qk_dim", 64)
        gcn_layers  = cfg.get("gcn_layers", 2)
        mlp_layers  = cfg.get("mlp_layers", 2)
        dropout     = cfg.get("dropout", 0.05)

        self.K = arg["data"]["num_classes"]
        self.kmeans_n = cfg.get("kmeans_n", 10)

        self.norm_bf = nn.LayerNorm(in_dim)
        self.layer = GCNRHLayer(in_dim, hidden_dim,
                                qk_dim, out_dim,
                                gcn_layers=gcn_layers,
                                mlp_layers=mlp_layers,
                                dropout=dropout)

        self.register_buffer("_B_o",      torch.empty(0), persistent=False)
        self.register_buffer("_cached_x", torch.empty(0), persistent=False)

    def prepare_input(self, data):
        """训练前调用一次：把增强特征缓存下来"""
        if self._cached_x.numel() == 0 or self._cached_x.device != data.x.device:
            with torch.no_grad():
                self._cached_x = build_augmented_x(data)

    def _prepare_B_o(self, data):
        if self._B_o.numel() == 0 or self._B_o.device != data.x.device:
            with torch.no_grad():
                A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                k = A.sum(dim=1)
                m = A.sum() / 2
                self._B_o = A - torch.outer(k, k) / (2 * m).clamp(min=1e-12)
        return self._B_o

    def forward(self, data):
        B_o = self._prepare_B_o(data)

        # ★ 判空用 numel()，不用 is not None
        x = self._cached_x if self._cached_x.numel() > 0 else build_augmented_x(data)
        x = self.norm_bf(x)
        C, mu, beta = self.layer(x, data.edge_index)
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
