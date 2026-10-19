# ============================================================
# src/ops/functional.py
# 通用数学运算与损失函数（全部为无状态纯函数）。
#
# 定位：
#   - 输入输出都是张量，不依赖具体模型
#   - 提供 softmax / mask 运算 / 常见损失
#   - 模型特有的损失逻辑放 models/<name>/loss.py，必要时调用这里的函数
# ============================================================

import torch
import torch.nn.functional as F


# ============================================================
# ① 概率运算
# ============================================================
def softmax(x, dim=-1):
    """沿指定维度的 softmax，等于 F.softmax"""
    return F.softmax(x, dim=dim)


def log_softmax(x, dim=-1):
    """数值稳定的 log-softmax（配合 NLLLoss 使用）"""
    return F.log_softmax(x, dim=dim)


def masked_softmax(x, mask, dim=-1, neg_inf=-1e9):
    """
    带掩码的 softmax：mask 为 False 的位置在 softmax 前被置为 -inf，
    使这些位置的输出概率为 0。

    参数:
        x    : (..., N) 的分数张量
        mask : (..., N) 的布尔张量，True 表示保留
    返回:
        与 x 同形状的概率张量
    """
    x = x.masked_fill(~mask, neg_inf)
    return F.softmax(x, dim=dim)


# ============================================================
# ② 掩码运算（图模型里最常用）
# ============================================================
def masked_select(x, mask):
    """按布尔掩码提取元素，mask 是 (N,) 张量，x 是 (N, ...) 张量"""
    return x[mask]


def masked_mean(x, mask, dim=0):
    """掩码位置上的均值；全 False 时返回 0，避免除零"""
    mask = mask.to(x.dtype)
    # 用 sum(mask) 作分母，避免直接 mean(empty) 的 NaN
    total = (x * mask.view(-1, *([1] * (x.dim() - 1)))).sum(dim=dim)
    count = mask.sum().clamp(min=1)
    return total / count


def masked_sum(x, mask, dim=0):
    """掩码位置上的和"""
    mask = mask.to(x.dtype)
    return (x * mask.view(-1, *([1] * (x.dim() - 1)))).sum(dim=dim)


# ============================================================
# ③ 分类损失
# ============================================================
def cross_entropy(pred, target, weight=None, label_smoothing=0.0):
    """
    交叉熵损失（分类）。

    参数:
        pred            : (N, C) 的 logits
        target          : (N,)   的类别索引
        weight          : (C,)   的类别权重，None 表示不加权
        label_smoothing : 标签平滑系数
    """
    return F.cross_entropy(pred, target,
                           weight=weight,
                           label_smoothing=label_smoothing)


def nll_loss(log_prob, target, weight=None):
    """负对数似然损失（输入需已过 log_softmax）"""
    return F.nll_loss(log_prob, target, weight=weight)


def binary_cross_entropy(pred, target, weight=None):
    """
    二分类交叉熵（输入为已过 sigmoid 的概率）。

    参数:
        pred   : (N,) 或 (N, 1) 的概率值
        target : 与 pred 同形状，取值 0 / 1
    """
    return F.binary_cross_entropy(pred, target.float(), weight=weight)


def binary_cross_entropy_with_logits(logits, target, weight=None):
    """二分类交叉熵（输入为未过 sigmoid 的 logits，数值更稳定）"""
    return F.binary_cross_entropy_with_logits(logits, target.float(), weight=weight)


# ============================================================
# ④ 回归损失
# ============================================================
def mse_loss(pred, target, reduction="mean"):
    """均方误差"""
    return F.mse_loss(pred, target, reduction=reduction)


def l1_loss(pred, target, reduction="mean"):
    """平均绝对误差"""
    return F.l1_loss(pred, target, reduction=reduction)


def smooth_l1_loss(pred, target, beta=1.0, reduction="mean"):
    """Huber 风格损失，异常值鲁棒"""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)


# ============================================================
# ⑤ 分布距离
# ============================================================
def kl_divergence(log_prob, target_prob, reduction="batchmean"):
    """
    KL 散度：KL(target || exp(log_prob))。
    注意输入 log_prob 是"对数概率"，不是概率本身。
    """
    return F.kl_div(log_prob, target_prob, reduction=reduction)


# ============================================================
# ⑥ 对比 / 度量学习损失
# ============================================================
def info_nce_loss(z1, z2, temperature=0.5):
    """
    InfoNCE 对比损失（SimCLR 风格）。
    输入两批配对的表示，把同索引视为正样本、其余为负样本。

    参数:
        z1, z2      : (N, D) 两组表示，z1[i] 与 z2[i] 是正对
        temperature : 温度系数，越小越尖锐
    """
    # 先做 L2 归一化，让点积等价于余弦相似度
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # (N, N) 相似度矩阵，除温度
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def triplet_loss(anchor, positive, negative, margin=1.0, p=2):
    """
    三元组损失：max(0, d(a,p) - d(a,n) + margin)
    """
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=p)


# ============================================================
# ⑦ 图相关：边上的 softmax / 消息聚合
# ============================================================
def scatter_softmax(src, index, num_nodes=None):
    """
    按 index 分组做 softmax，用于边权重归一化（图注意力）。
    等价于 PyG 的 scatter_softmax，这里做一层薄封装方便统一调用。

    参数:
        src       : (E,) 每条边的分数
        index     : (E,) 每条边属于哪个目标节点
        num_nodes : 目标节点数，None 时从 index 推断
    """
    # 先减每组最大值，避免 exp 溢出
    if num_nodes is None:
        num_nodes = int(index.max()) + 1

    # 用 scatter 求每组最大值
    max_per_group = torch.full((num_nodes,), float("-inf"),
                               device=src.device, dtype=src.dtype)
    max_per_group = max_per_group.scatter_reduce(
        0, index, src, reduce="amax", include_self=True
    )
    src = src - max_per_group[index]

    exp_src = torch.exp(src)
    # 求和
    denom = torch.zeros(num_nodes, device=src.device, dtype=src.dtype)
    denom = denom.scatter_add(0, index, exp_src)
    return exp_src / denom[index].clamp(min=1e-12)