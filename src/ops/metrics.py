# ============================================================
# src/ops/metrics.py
# 评估指标（全部为无状态纯函数）。
#
# 输入约定：
#   - 分类的 output 既可以是 (N, C) 的 logits，也可以是 (N,) 的类别索引
#   - 内部统一先转成类别索引，再做比较
# ============================================================

import torch


# ============================================================
# 内部工具：把 logits 转成预测类别索引
# ============================================================
def _to_pred(output):
    """(N, C) → (N,)；已是 (N,) 则原样返回"""
    pred = output.argmax(dim=-1) if output.dim() > 1 else output
    return pred.view(-1)


def _to_target(target):
    """统一为 1D 的 long 张量"""
    return target.view(-1).long()


# ============================================================
# ① 分类：准确率
# ============================================================
def accuracy(output, target):
    """分类准确率，返回 0~1 的 float"""
    pred = _to_pred(output)
    target = _to_target(target)
    return (pred == target).sum().item() / max(target.numel(), 1)


def top_k_accuracy(output, target, k=5):
    """
    Top-K 准确率：正确类别出现在前 K 个预测里即算对。
    仅对 (N, C) 的 logits 有效。
    """
    assert output.dim() > 1, "top_k_accuracy 需要 (N, C) 的 logits"
    target = _to_target(target)

    # 取 top-k 的类别索引，看 target 是否在其中
    _, top_k_pred = output.topk(k, dim=-1)
    correct = (top_k_pred == target.unsqueeze(-1)).any(dim=-1)
    return correct.sum().item() / max(target.numel(), 1)


# ============================================================
# ② 分类：混淆矩阵与 P/R/F1
# ============================================================
def confusion_matrix(output, target, num_classes=None):
    """
    返回 (C, C) 的混淆矩阵，行是真实类别，列是预测类别。

    参数:
        num_classes : 不传时从 target / pred 的最大值推断
    """
    pred   = _to_pred(output)
    target = _to_target(target)

    if num_classes is None:
        num_classes = int(max(pred.max(), target.max()).item()) + 1

    # 用线性索引一次性累加：idx = target * C + pred
    idx = target * num_classes + pred
    cm = torch.zeros(num_classes * num_classes, dtype=torch.long)
    cm = cm.scatter_add(0, idx.cpu(), torch.ones_like(idx.cpu()))
    return cm.view(num_classes, num_classes)


def precision_recall_f1(output, target, num_classes=None, average="macro"):
    """
    计算 precision / recall / f1。

    参数:
        average : "macro"（各类平均） / "micro"（全局统计） / "weighted"（按类频加权）
    返回:
        dict: {"precision": float, "recall": float, "f1": float}
    """
    cm = confusion_matrix(output, target, num_classes).float()   # (C, C)
    C  = cm.size(0)

    # TP / FP / FN 按类
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp          # 列和 - 对角
    fn = cm.sum(dim=1) - tp          # 行和 - 对角
    support = cm.sum(dim=1)          # 每类真实样本数

    # 每类的 P / R / F1，分母为 0 时置 0
    prec = tp / (tp + fp).clamp(min=1e-12)
    rec  = tp / (tp + fn).clamp(min=1e-12)
    f1   = 2 * prec * rec / (prec + rec).clamp(min=1e-12)

    if average == "macro":
        p, r, f = prec.mean(), rec.mean(), f1.mean()
    elif average == "weighted":
        w = support / support.sum().clamp(min=1e-12)
        p, r, f = (prec * w).sum(), (rec * w).sum(), (f1 * w).sum()
    elif average == "micro":
        # 微平均等价于全局 accuracy（多分类）
        p = r = f = tp.sum() / cm.sum().clamp(min=1e-12)
    else:
        raise ValueError(f"未知 average: {average}，可选 macro / weighted / micro")

    return {"precision": float(p), "recall": float(r), "f1": float(f)}


# ============================================================
# ③ 回归指标
# ============================================================
def mse(pred, target):
    """均方误差"""
    pred   = pred.view(-1).float()
    target = target.view(-1).float()
    return ((pred - target) ** 2).mean().item()


def mae(pred, target):
    """平均绝对误差"""
    pred   = pred.view(-1).float()
    target = target.view(-1).float()
    return (pred - target).abs().mean().item()


def r2_score(pred, target):
    """
    决定系数 R²：1 表示完美预测，0 表示等同均值预测，可为负。
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum().clamp(min=1e-12)
    return (1 - ss_res / ss_tot).item()


# ============================================================
# ④ 二分类指标（AUC 等，不依赖 sklearn 的轻量实现）
# ============================================================
def auc_roc(scores, target):
    """
    二分类 ROC-AUC 的秩和法实现。

    参数:
        scores : (N,) 正类分数（越大越像正类）
        target : (N,) 0/1 标签
    """
    scores = scores.view(-1).float()
    target = target.view(-1).long()

    pos = scores[target == 1]
    neg = scores[target == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")

    # 比较所有 (正, 负) 对的分数：正 > 负 计 1，正 == 负 计 0.5
    cmp = (pos.unsqueeze(1) > neg.unsqueeze(0)).float() \
        + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float()
    return cmp.mean().item()