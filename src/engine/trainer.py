# ============================================================
# src/engine/trainer.py
# 训练主循环：epoch 迭代 → 训练 → 验证 → 记录 → 保存 → 早停。
#
# 设计：
#   - 单图训练（PyG Data 整图），用 mask 区分 train/val/test
#   - Trainer 不 import 任何具体模型；模型通过参数传入
#   - 损失签名 criterion(pred, target) -> scalar
# ============================================================

import torch
import os

from src.utils.registry import build_model, build_loss
from src.utils.seed import set_seed
from src.config import save_yaml
from src.engine.logger import Logger
from src.engine.checkpoint import Checkpoint
from src.engine.evaluator import evaluate
from src.utils.device import get_device


class Trainer:
    def __init__(self, arg, model, data, criterion, optimizer, ckpt, logger):
        self.arg        = arg
        self.model      = model
        self.data       = data.to(get_device(arg))   # 数据搬到设备
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.ckpt       = ckpt
        self.logger     = logger
        self.device     = get_device(arg)

        # 模型也搬过去（在 fit 之前完成）
        self.model.to(self.device)
        self.logger.info(f"使用设备: {self.device}")

        # 从 arg 读取训练配置
        self.epochs     = arg["train"]["epochs"]
        self.grad_clip  = arg["train"]["grad_clip"]
        self.early_stop = arg["train"]["early_stop"]
        self.log_every  = arg["train"]["log_every"]
        self.monitor    = arg["save"]["monitor"]     # 如 "val_acc"
        self.mode       = arg["save"]["mode"]        # "max" / "min"

        # 早停状态
        self.best_value    = float("-inf") if self.mode == "max" else float("inf")
        self.patience_left = self.early_stop or 0

    # ---------------------------------------------------------
    def fit(self):
        """训练主循环"""
        last_metrics = {}

        for epoch in range(1, self.epochs + 1):
            train_loss, train_extras = self._train_one_epoch()

            metrics = {"train_loss": train_loss}
            for k, v in train_extras.items():
                metrics[f"train_{k}"] = v

            for split in ("train", "val"):
                m = self._evaluate_split(split)
                for k, v in m.items():
                    metrics[f"{split}_{k}"] = v
            last_metrics = metrics

            # 记录
            if epoch % self.log_every == 0:
                self.logger.log_epoch(epoch, metrics)

            # 保存 last（用于 resume）
            self.ckpt.save_last(epoch, self.model, self.optimizer, metrics)

            # 判断是否 best / 早停
            if self.monitor not in metrics:
                self.logger.warning(f"监控指标 '{self.monitor}' 不在本轮指标中：{list(metrics)}")
                continue
            v = metrics[self.monitor]
            if self._is_better(v):
                self.best_value    = v
                self.patience_left = self.early_stop or 0
                self.ckpt.save_best(epoch, self.model, self.optimizer, metrics, value=metrics[self.monitor])
                self.logger.info(f"  ↳ 保存 best ({self.monitor}={v:.4f})")
            else:
                if self.early_stop:
                    self.patience_left -= 1
                    if self.patience_left <= 0:
                        self.logger.info(
                            f"早停触发于 epoch {epoch}，最佳 {self.monitor}={self.best_value:.4f}"
                        )
                        break

        return last_metrics

    # ---------------------------------------------------------
    def _train_one_epoch(self):
        """一轮训练：forward → loss → backward → step"""
        self.model.train()
        self.optimizer.zero_grad()

        # ★ 模型输出原样传给损失
        outputs = self.model(self.data)
        loss, extras = self._compute_loss(outputs)

        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss), extras  # 返回 (loss, 附加指标)

    # ---------------------------------------------------------
    def test(self, ckpt_path=None):
        """在 test split 上做最终评估；ckpt_path 非空则先加载权重"""
        if ckpt_path is not None:
            self.ckpt.load(ckpt_path, self.model, None, self.device)
            self.logger.info(f"加载权重进行测试: {ckpt_path}")

            # ★ 同样走统一入口
        metrics = self._evaluate_split("test")
        self.logger.info(f"Test: {metrics}")
        return metrics

    # ---------------------------------------------------------
    def _is_better(self, v):
        return v > self.best_value if self.mode == "max" else v < self.best_value


    def _evaluate_split(self, split):
        """
        评估指定 split。优先使用模型自己定义的 evaluate（如果有），
        否则走默认的（loss + acc）。
        """
        if hasattr(self.model, "evaluate") and callable(self.model.evaluate):
            # 模型特定的评估
            return self.model.evaluate(self.data, split=split)

        # 默认路径：loss + acc
        mask = getattr(self.data, f"{split}_mask")
        return evaluate(self.model, self.data, mask, self.criterion)


    def _compute_loss(self, outputs):
        """
        调用损失，兼容两种签名和返回值：
          签名 A（新）: criterion(outputs, data) -> loss 或 (loss, extras)
          签名 B（旧）: criterion(pred, target)  -> loss

        通过对损失 forward 参数个数的检查来选择调用方式。
        """
        import inspect
        sig = inspect.signature(self.criterion.forward)
        nargs = len([
            p for p in sig.parameters.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ])

        if nargs == 1:
            # 只用 outputs
            result = self.criterion(outputs)
        elif nargs >= 2:
            # 新签名：criterion(outputs, data)
            result = self.criterion(outputs, self.data)
        else:
            # 旧签名：criterion(pred, target)
            # 假设 outputs 是 logits；分类任务用 train_mask 筛选
            pred = outputs["logits"] if isinstance(outputs, dict) else outputs
            target = self.data.y
            mask = self.data.train_mask
            result = self.criterion(pred[mask], target[mask])

        # 解析返回值：loss 或 (loss, extras)
        if isinstance(result, tuple):
            loss, extras = result[0], result[1]
            if not isinstance(extras, dict):
                extras = {f"extra_{i}": v for i, v in enumerate(result[1:])}
        else:
            loss, extras = result, {}

        return loss, extras