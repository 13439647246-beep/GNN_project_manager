# ============================================================
# src/engine/logger.py
# 统一日志：控制台 + 文件 + CSV 指标。
# run_dir 由外部传入（runner/run_layout 决定）；不传则保留旧逻辑。
# ============================================================

import csv
import logging
import os
import uuid
from datetime import datetime


class Logger:
    def __init__(self, arg, run_dir=None):
        self.arg = arg

        # 优先用外部传入的 run_dir；否则保留旧逻辑（向后兼容）
        if run_dir is not None:
            self.run_dir = str(run_dir)
        else:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = arg["exp"]["name"] or arg["model"]["name"] or "exp"
            self.run_dir = os.path.join(
                arg["exp"]["output_dir"],
                arg["model"]["name"],
                exp_name,
                ts,
            )
        os.makedirs(self.run_dir, exist_ok=True)

        # ★ logger name 必须唯一，否则 grid 同秒多个 Logger 会串 handler
        name = f"mlzoo.{uuid.uuid4().hex}"
        self._logger = logging.getLogger(name)
        self._logger.setLevel(arg["log"]["level"])
        self._logger.handlers.clear()
        self._logger.propagate = False

        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler(); ch.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(self.run_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

        self._csv_path   = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file   = None
        self._csv_writer = None
        self._csv_header = None

    def info(self, msg):    self._logger.info(msg)
    def warning(self, msg): self._logger.warning(msg)

    def log_epoch(self, epoch, metrics: dict):
        if self._csv_writer is None:
            self._csv_header = ["epoch"] + list(metrics.keys())
            self._csv_file   = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self._csv_header)

        row = [epoch] + [metrics[k] for k in self._csv_header[1:]]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        brief = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self._logger.info(f"Epoch {epoch:>4d}  {brief}")

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)