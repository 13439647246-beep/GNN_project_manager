# ============================================================
# models/gcnrh/loss.py
# GCN-RH 损失：重构损失 + KL 散度。
#
# 新签名：forward(outputs, data) -> (loss, extras)
#   - outputs : 模型 forward 返回的 dict
#   - data    : 原始 PyG Data（备用，你这个损失其实只需要 outputs 里的 B_o）
#   - 返回总损失 + 一个 dict 记录各分量（Trainer 会写进 metrics.csv）
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import register_loss


@register_loss("GCNRH_loss")
def build_loss(arg):
    return GCNRHLoss(arg)


class GCNRHLoss(nn.Module):
    def __init__(self, arg):
        super().__init__()
        cfg = arg["loss"].get("kwargs", {}) or {}
        self.lambda_kl = cfg.get("lambda_kl", 0.5)

        self.log_temp = nn.Parameter(torch.tensor(0.0))  # exp(0)=1

    def forward(self, outputs):
        B_o = outputs["B_o"]
        C = outputs["C"]  # ★ 用 C，不用 mu
        mu = outputs["mu"]
        beta = outputs["beta"]

        target = (B_o > 0).float()
        Bp = torch.exp(self.log_temp) * (C @ C.T)  # ★ 用 C

        n = B_o.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=B_o.device)
        logits = Bp[mask]
        labels = target[mask]

        pos = labels > 0.5
        neg = ~pos
        loss_pos = F.binary_cross_entropy_with_logits(logits[pos], labels[pos], reduction="mean") if pos.any() else 0.0
        loss_neg = F.binary_cross_entropy_with_logits(logits[neg], labels[neg], reduction="mean")
        L_G = loss_pos + loss_neg

        eps = 1e-10
        L_C = 0.5 * torch.mean(mu.pow(2) + beta.pow(2) - 2 * torch.log(beta + eps) - 1)
        L = self.lambda_kl * L_C + L_G
        return L, {"L_G": float(L_G.detach()), "L_C": float(L_C.detach())}