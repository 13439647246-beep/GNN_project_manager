# ============================================================
# src/ops/linalg.py
# 线性代数相关运算：相似度、距离、归一化、图谱运算。
# 全部为无状态纯函数。
# ============================================================

import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# ============================================================
# ① 向量归一化
# ============================================================
def l2_normalize(x, dim=-1, eps=1e-12):
    """
    L2 归一化：把向量缩放到单位长度。
    常在对表示做相似度计算前调用。

    参数:
        x   : 任意形状的张量
        dim : 沿哪个维度归一化
        eps : 防止除零
    """
    norm = x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    return x / norm


# ============================================================
# ② 相似度
# ============================================================
def cosine_similarity(x, y, dim=-1):
    """
    余弦相似度：cos(x, y)。
    输入形状需能广播（比如 (N, D) 与 (D,)）。
    """
    x = l2_normalize(x, dim=dim)
    y = l2_normalize(y, dim=dim)
    return (x * y).sum(dim=dim)


def pairwise_cosine_similarity(x):
    """
    批量余弦相似度矩阵。输入 (N, D)，输出 (N, N)。
    对角线上是自己与自己的相似度（恒为 1）。
    """
    x = l2_normalize(x, dim=-1)
    return x @ x.t()


# ============================================================
# ③ 距离
# ============================================================
def pairwise_euclidean_distances(x, y=None):
    """
    批量欧氏距离矩阵。
        - 只给 x：(N, D) → (N, N)
        - 给 x, y：(N, D), (M, D) → (N, M)

    用 (a-b)^2 = a^2 + b^2 - 2ab 展开，避免构造三维差分。
    """
    if y is None:
        y = x

    # 平方和
    x2 = (x ** 2).sum(dim=-1, keepdim=True)          # (N, 1)
    y2 = (y ** 2).sum(dim=-1, keepdim=True).t()      # (1, M)

    dist2 = x2 + y2 - 2 * x @ y.t()
    # 数值误差可能产生微负，clamp 一下
    return dist2.clamp(min=0).sqrt()


# ============================================================
# ④ Top-K
# ============================================================
def top_k_indices(x, k, dim=-1, largest=True, sorted=True):
    """
    返回 top-k 的下标（只要下标，不要值）。
    常用于检索评估、负采样。
    """
    return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted).indices


# ============================================================
# ⑤ 图相关：邻接矩阵的对称归一化
# ============================================================
def symmetric_normalize_adj(adj):
    """
    对称归一化邻接矩阵：Â = D^{-1/2} A D^{-1/2}。
    这是 GCN 论文的核心公式，输入输出都是稠密矩阵。

    参数:
        adj : (N, N) 的稠密邻接矩阵
    返回:
        (N, N) 的归一化邻接矩阵

    注意：PyG 的 GCNConv 已内置稀疏版本的归一化，这里提供稠密版本，
         主要给需要手动构造矩阵、做谱分析等场景使用。
    """
    deg = adj.sum(dim=-1)                  # (N,) 度数
    # D^{-1/2}，度数为 0 的位置置 0
    d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    d_mat = torch.diag(d_inv_sqrt)         # (N, N)
    return d_mat @ adj @ d_mat


def add_self_loops_dense(adj):
    """给稠密邻接矩阵加自环：A + I"""
    n = adj.size(0)
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    return adj + eye


# ============================================================
# 计算模块度
# ============================================================
def modularity_matrix(edge_index, num_nodes=None):
    """
    由 edge_index 计算模块度矩阵 B = A - k k^T / (2m)。

    参数:
        edge_index : (2, E) 的 LongTensor（torch 张量）
        num_nodes  : 节点数，None 时从 edge_index 推断

    返回:
        (N, N) 的 float 张量
    """
    # ★ 强制转成 torch 张量（兼容误传 numpy / list 的情况）
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()

    device = edge_index.device

    # 节点数：从 edge_index 推断
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0

    # 空图直接返回全 0
    if num_nodes == 0:
        return torch.zeros(0, 0, device=device)

    # 对称化 + 去自环
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]
    u, v = edge_index[0], edge_index[1]

    # 度向量 k
    ones = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    k = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    k = k.scatter_add(0, u, ones)
    k = k.scatter_add(0, v, ones)

    m2 = k.sum()                 # 2m
    if m2.item() == 0:
        return torch.zeros(num_nodes, num_nodes, device=device)

    # 邻接矩阵 A
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A = A.index_put((u, v), torch.ones_like(u, dtype=torch.float32), accumulate=True)

    # B = A - k k^T / (2m)
    return A - torch.outer(k, k) / m2


def modularity_score(edge_index, labels, num_nodes=None):
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()
    device = edge_index.device

    n_from_edges = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    n_from_labels = len(labels) if labels is not None else 0
    num_nodes = num_nodes or max(n_from_edges, n_from_labels)

    if labels is not None and not isinstance(labels, torch.Tensor):
        if hasattr(labels, "values"):
            labels = labels.values
        labels = torch.as_tensor(labels)
    labels = labels.to(device=device, dtype=torch.long)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]
    u, v = edge_index[0], edge_index[1]

    # 度：对称化后只对 u 做一次 scatter_add
    ones = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    k = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    k = k.scatter_add(0, u, ones)          # ★ 只做一次
    m2 = k.sum()                           # = 2m
    if m2.item() == 0:
        return 0.0

    # 邻接矩阵：用 sparse 或去重后置 1，避免重复边累加
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A = A.index_put((u, v), torch.ones_like(u, dtype=torch.float32), accumulate=True)
    A = (A > 0).float()                    # ★ 重复边只算 1 次

    B = A - torch.outer(k, k) / m2
    same = (labels[:, None] == labels[None, :]).to(torch.float32)
    Q = (B * same).sum() / m2
    return float(Q.item())









