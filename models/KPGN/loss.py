import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import register_loss


@register_loss("KPGN_loss")
def build_loss(arg):
    return KPGNLoss(arg)


class KPGNLoss(nn.Module):
    def __init__(self, arg):
        super().__init__()
        cfg = arg["loss"].get("kwargs", {}) or {}
        self.alpha = cfg.get("alpha", 1.0)

    def forward(self, outputs, data):
        num_nodes = data.num_nodes
        labels = data.y  # (N,) 全图标签
        idx_train = outputs["idx_train"]
        logits_train = outputs["output_train"]  # 从 outputs 里拿，无需依赖 data
        logits_all = outputs["output_all"]
        edge_index = outputs["edge_index"]
        loss_ce = F.cross_entropy(logits_train, labels[idx_train])
        loss_q = modularity_loss(logits_all, edge_index, num_nodes)
        loss = loss_ce + self.alpha * loss_q
        return loss


def modularity_loss(logits, edge_index, num_nodes):
    """
    用 softmax 后的类别概率近似计算模块度损失。
    logits: [N, C]
    edge_index: [2, E] 的 LongTensor，表示所有边
    """
    probs = F.softmax(logits, dim=1)   # [N, C]
    deg = torch.zeros(num_nodes, device=logits.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=logits.device))

    m = edge_index.size(1) / 2.0       # 无向图，除以 2
    delta = (probs[edge_index[0]] * probs[edge_index[1]]).sum(dim=1)  # [E]

    # 简化版：只算第一项
    Q = (delta / (2 * m)).sum() \
        - ((deg[edge_index[0]] * deg[edge_index[1]]) / (4 * m * m)).sum()

    return 1 - Q