# 全局随机种子固定（torch/numpy/random）
# ============================================================
# src/utils/seed.py
# 全局随机种子固定：保证实验可复现
# ============================================================
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)