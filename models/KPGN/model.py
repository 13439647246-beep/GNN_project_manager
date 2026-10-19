import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from src.utils.registry import register_model
from models.KPGN.layer import KPGNLayer
from models.KPGN.prepare import (normalize_adj, build_adjacency_dict_sparse, similary_sampling,
                                 torch_adj_to_nx, bs_max_kplex, runMain, seeds_to_labels)


@register_model("KPGN")
def build_model(arg):
    """注册的构建函数：arg → 模型实例"""
    return KPGN(arg)


def prepare_adj(data):
    n = data.num_nodes
    edge_weight = torch.ones(data.num_edges, dtype=torch.float32, device=data.edge_index.device)
    A = torch.sparse_coo_tensor(
        data.edge_index, edge_weight, size=(n, n)
    ).coalesce()
    return normalize_adj(A)

def prepare_idx_train(data, k, lamda):
    n = data.num_nodes
    edge_weight = torch.ones(data.num_edges, dtype=torch.float32, device=data.edge_index.device)
    A = torch.sparse_coo_tensor(
        data.edge_index, edge_weight, size=(n, n)
    ).coalesce()
    G_nx = torch_adj_to_nx(A)
    common = runMain(k, lamda, G_nx)
    seed_labels, c, seed_mask = seeds_to_labels(common, n)
    idx_train = torch.nonzero(seed_mask).view(-1)
    return idx_train


class KPGN(nn.Module):
    def __init__(self, arg):
        super().__init__()
        cfg = arg["model"]["kwargs"]
        in_dim = arg["data"]["num_features"]
        out_dim = arg["data"]["num_classes"]
        hidden_dims = cfg.get("hidden_dims", [256])
        self.k = cfg.get("k", 2)
        self.lamda = cfg.get("lamda", 5)
        self.num_neighbors_list = cfg.get("num_neighbors_list", [10,5])
        self.num_nodes = arg["data"]["num_nodes"]

        self.layer = KPGNLayer(in_dim, hidden_dims, out_dim, self.num_neighbors_list, self.num_nodes)

        self.register_buffer("_A_hat", torch.empty(0), persistent=False)
        self.register_buffer("_idx_train", torch.empty(0), persistent=False)
        self._nei_adj_dict = None

    def forward(self, data):
        # ★ 判空用 numel()，不用 is not None
        labels = data.y
        class_num = data.y.unique().numel()

        # A_hat：第一次算，之后复用
        if self._A_hat.numel() == 0 or self._A_hat.device != data.x.device:
            self._A_hat = prepare_adj(data)
        adj = self._A_hat

        # idx_train：第一次算，之后复用
        if self._idx_train.numel() == 0:
            self._idx_train = prepare_idx_train(data, self.k, self.lamda).to(data.y.device)
        idx_train = self._idx_train

        valid_train = [i for i in idx_train.tolist() if 0 <= labels[i] < class_num]
        idx_train = torch.LongTensor(valid_train)
        adj_dict = build_adjacency_dict_sparse(adj, True)
        all_nodes = torch.arange(int(self.num_nodes))
        edge_index = adjacency_dict_to_edge_index(adj_dict, self.num_nodes).to(data.y.device)

        # nei_adj_dict：普通属性缓存
        if self._nei_adj_dict is None:
            self._nei_adj_dict = similary_sampling(adj_dict, self.num_nodes)
        nei_adj_dict = self._nei_adj_dict

        # ---- 前向 1：训练节点，用于交叉熵 ----
        nfl_train = build_node_features_list(
            center_nodes=idx_train,
            nei_adj_dict=nei_adj_dict,
            num_neighbors_list=self.num_neighbors_list,
            features=data.x
        )
        _, logits_train = self.layer(nfl_train)

        # ---- 前向 2：全图，用于模块度 ----
        nfl_all = build_node_features_list(
            center_nodes=all_nodes,
            nei_adj_dict=nei_adj_dict,
            num_neighbors_list=self.num_neighbors_list,
            features=data.x
        )
        _, logits_all = self.layer(nfl_all)

        return {"edge_index": edge_index, "idx_train": idx_train,
                "output_train": logits_train, "output_all": logits_all}

    # =========================================================
    # 模型特定的评估：Trainer 检测到此方法则优先使用
    # =========================================================
    @torch.no_grad()
    def evaluate(self, data, split="test"):
        """
        在指定 split 上评估节点分类指标。

        参数:
            data  : PyG Data
            split : "train" / "val" / "test"
        返回:
            dict: {"loss": float, "acc": float}
        """
        self.eval()
        outputs = self(data)
        logits_all = outputs["output_all"]  # (N, C)

        # 按 split 选择节点掩码
        mask = getattr(data, f"{split}_mask")
        y = data.y

        # 只对掩码位置算指标
        logits_sub = logits_all[mask]
        y_sub = y[mask]

        if y_sub.numel() == 0:
            return {"loss": float("nan"), "acc": float("nan")}

        loss = F.cross_entropy(logits_sub, y_sub)
        pred = logits_sub.argmax(dim=-1)
        acc = (pred == y_sub).float().mean().item()

        return {"loss": float(loss), "acc": float(acc)}



def adjacency_dict_to_edge_index(adjacency_dict, num_nodes=None):
    edges = []
    for u, neighbors in adjacency_dict.items():
        for v in neighbors:
            edges.append((u, v))
    if num_nodes is None:
        num_nodes = max(max(u, v) for u, v in edges) + 1
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def build_node_features_list(center_nodes, nei_adj_dict, num_neighbors_list, features):
    """
    center_nodes: torch.LongTensor，中心节点索引
    nei_adj_dict: 预先计算好的邻居字典 {node: [top-t neighbors]}
    num_neighbors_list: 每层采样数量，例如 [20, 10]
    features: torch.Tensor [N, d]

    返回：node_features_list
    """
    if isinstance(center_nodes, torch.Tensor):
        center_nodes_list = center_nodes.tolist()
    else:
        center_nodes_list = list(center_nodes)

    feature_list = [features[center_nodes]]
    current_nodes = center_nodes_list

    for num_sample in num_neighbors_list:
        sampled = []
        for u in current_nodes:
            nbrs = nei_adj_dict.get(u, [])
            if len(nbrs) == 0:
                chosen = [u] * num_sample
            elif len(nbrs) <= num_sample:
                chosen = nbrs + [random.choice(nbrs) for _ in range(num_sample - len(nbrs))]
            else:
                chosen = random.sample(nbrs, num_sample)
            sampled.extend(chosen)

        sampled_nodes = torch.LongTensor(sampled)
        feature_list.append(features[sampled_nodes])
        current_nodes = sampled_nodes.tolist()

    return feature_list