# ============================================================
# src/engine/evaluator.py
# 在给定 mask 上评估模型：返回 loss 和 acc。
# 因为 Cora 这类图是单图 + 掩码，不需要 DataLoader。
# ============================================================

import torch

from src.ops.metrics import accuracy


def evaluate(model, data, mask, criterion):
    """
    参数:
        model     : 模型
        data      : PyG Data 对象（已搬到 device）
        mask      : 用于评估的布尔掩码（如 data.val_mask）
        criterion : 损失函数，签名 (pred, target) -> scalar
    返回:
        {"loss": float, "acc": float}
    """
    model.eval()
    with torch.no_grad():
        out  = model(data)
        loss = criterion(out[mask], data.y[mask])
        acc  = accuracy(out[mask], data.y[mask])
    return {"loss": float(loss), "acc": float(acc)}