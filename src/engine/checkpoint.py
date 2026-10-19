# src/engine/checkpoint.py
from pathlib import Path

import torch


class Checkpoint:
    def __init__(self, arg, logger):
        self.arg = arg
        self.logger = logger
        self.dir = Path(logger.run_dir) / "ckpt"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.dir / "best.pt"
        self.last_path = self.dir / "last.pt"
        self.monitor = arg["save"]["monitor"]
        self.mode = arg["save"].get("mode", "max")
        self.best_value = None
        self.best_epoch = None

    def _pack(self, model, optimizer, epoch, metrics, best_value):
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "monitor": self.monitor,
            "best_value": best_value,
            "metrics": metrics or {},
            "arg": self.arg,
        }

    def save_last(self, epoch, model, optimizer=None, metrics=None):
        torch.save(self._pack(model, optimizer, epoch, metrics, self.best_value), self.last_path)

    def save_best(self, epoch, model, optimizer=None, metrics=None, value=None):
        if value is None:
            value = (metrics or {})[self.monitor]
        better = self.best_value is None or (
            value > self.best_value if self.mode == "max" else value < self.best_value
        )
        if better:
            self.best_value = value
            self.best_epoch = epoch
            torch.save(self._pack(model, optimizer, epoch, metrics, value), self.best_path)
            return True
        return False

    def load(self, path, model, optimizer=None, device="cpu"):
        obj = torch.load(path, map_location=device)
        model.load_state_dict(obj["model"])
        if optimizer is not None and obj.get("optimizer"):
            optimizer.load_state_dict(obj["optimizer"])
        return obj


def load_best_model(ckpt_path, device="cpu"):
    """从自包含 best.pt 直接恢复模型，不依赖 configs/ 或 runs/ 其他目录。"""
    from src.utils.registry import build_model

    obj = torch.load(ckpt_path, map_location=device)
    arg = obj["arg"]
    model = build_model(arg).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    return model, arg, obj