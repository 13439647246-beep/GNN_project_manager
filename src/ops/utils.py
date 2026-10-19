# ============================================================
# src/ops/utils.py
# 杂项数值 / 类型工具（无状态纯函数）。
#
# 定位：放"哪一类的归不进"的小工具——类型转换、one-hot、参数计数。
# 注意：随机种子固定属于 src/utils/seed.py，不在这里重复。
# ============================================================

import numpy as np
import torch


# ============================================================
# ① 类型转换：numpy ↔ tensor，标量转换
# ============================================================
def to_tensor(x, device=None, dtype=None):
    """numpy / list / 标量 → torch.Tensor；已经是张量则按需搬移 / 转型"""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if dtype is not None:
        x = x.to(dtype)
    if device is not None:
        x = x.to(device)
    return x


def to_numpy(x):
    """torch.Tensor / list / 标量 → numpy.ndarray（自动脱离计算图与 GPU）"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_scalar(x):
    """任意张量 / numpy / 标量 → Python 原生数值"""
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x.tolist()
    return x


# ============================================================
# ② one-hot
# ============================================================
def to_one_hot(labels, num_classes, dtype=torch.float32):
    """
    把类别索引转成 one-hot。

    参数:
        labels      : (N,) 的 int 张量或 list
        num_classes : 类别数
        dtype       : 输出类型，通常 float32
    返回:
        (N, num_classes) 的 one-hot 张量
    """
    labels = to_tensor(labels).long().view(-1)
    return torch.nn.functional.one_hot(labels, num_classes).to(dtype)


def from_one_hot(one_hot):
    """(N, C) one-hot → (N,) 类别索引"""
    return to_tensor(one_hot).argmax(dim=-1)


# ============================================================
# ③ 模型参数统计
# ============================================================
def count_parameters(model, trainable_only=True):
    """
    统计模型参数量。

    参数:
        model          : nn.Module
        trainable_only : True 只统计 requires_grad=True 的参数
    返回:
        int，参数总数
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model):
    """打印每一层的名字、形状、参数量；末尾给出总数"""
    print("=" * 78)
    print(f"{'Module':<48s}{'Shape':<20s}{'Params':>10s}")
    print("-" * 78)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        shape = "x".join(str(s) for s in p.shape)
        print(f"{name:<48s}{shape:<20s}{n:>10,d}")
    print("-" * 78)
    print(f"{'Total':<68s}{total:>10,d}")
    print("=" * 78)


# ============================================================
# ④ 数值稳定工具
# ============================================================
def safe_divide(a, b, eps=1e-12):
    """安全除法：b 为 0 时返回 0，避免 NaN"""
    return a / b.clamp(min=eps) if torch.is_tensor(b) else a / max(b, eps)


def log1p_exp(x):
    """log(1 + exp(x)) 的数值稳定实现，= softplus(x)"""
    return torch.nn.functional.softplus(x)


# ============================================================
# ⑤ 结构操作
# ============================================================
def flatten_dict(d, parent_key="", sep="."):
    """
    把嵌套 dict 展平为一层，键用 sep 连接。
    例如 {"a": {"b": 1}} → {"a.b": 1}
    与 src/config/arg_io.py 里的点号路径对应，便于日志打印。
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def merge_dicts(dicts, sep="."):
    """把多个嵌套 dict 合并成一个展平的 dict（后者覆盖前者）"""
    out = {}
    for d in dicts:
        out.update(flatten_dict(d, sep=sep))
    return out