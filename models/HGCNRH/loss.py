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


@register_loss("HGCNRH_loss")
def build_loss(arg):
    return HGCNRHLoss(arg)


class HGCNRHLoss(nn.Module):
    def __init__(self, arg):
        super().__init__()
        cfg = arg["loss"].get("kwargs", {}) or {}
        # ★ target 值，warm-up 结束后用这个
        self.lambda_kl_target = cfg.get("lambda_kl", 0.1)
        # ★ warm-up 步数，约等于 epoch 数；整图训练一 epoch 一次 forward
        self.warmup_steps = cfg.get("kl_warmup_steps", 50)
        self._train_step = 0          # 只统计训练前向

        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, outputs):
        B_o  = outputs["B_o"]
        C    = outputs["C"]
        mu   = outputs["mu"]
        beta = outputs["beta"]

        target = (B_o > 0).float()
        k = C.size(-1)

        Bp = torch.exp(self.log_temp) * (C @ C.T) / (k ** 0.5)

        n      = B_o.size(0)
        mask   = ~torch.eye(n, dtype=torch.bool, device=B_o.device)
        logits = Bp[mask]
        labels = target[mask]

        pos = labels > 0.5
        neg = ~pos
        loss_pos = (
            F.binary_cross_entropy_with_logits(logits[pos], labels[pos], reduction="mean")
            if pos.any() else torch.tensor(0.0, device=B_o.device)
        )
        loss_neg = F.binary_cross_entropy_with_logits(logits[neg], labels[neg], reduction="mean")
        L_G = loss_pos + loss_neg

        # ★ KL warm-up：只在训练前向（grad 开启）时递增
        if torch.is_grad_enabled():
            self._train_step += 1
            progress = min(1.0, self._train_step / max(1, self.warmup_steps))
            lambda_kl = self.lambda_kl_target * progress
        else:
            lambda_kl = self.lambda_kl_target   # 评估时用满值，只影响日志，不影响训练

        eps = 1e-10
        L_C = 0.5 * torch.mean(mu.pow(2) + beta.pow(2) - 2 * torch.log(beta + eps) - 1)
        L = lambda_kl * L_C + L_G
        return L, {"L_G": float(L_G.detach()), "L_C": float(L_C.detach()), "lambda_kl": float(lambda_kl)}