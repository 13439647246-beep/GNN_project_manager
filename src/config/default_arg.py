# ============================================================
# src/config/default_arg.py
# 全局默认 arg 字典：所有模型读同一套 key 结构。
#
# 各模型的 yaml 文件在此基础上做覆盖；模型专属超参统一放在
# model.kwargs / loss.kwargs 里，不随意新增顶层字段，
# 这样 engine / grid / logger 都能用统一路径访问。
# ============================================================

DEFAULT_ARG = {
    # ---- 模型 ----
    "model": {
        "name":   None,       # 模型名（对应 models/ 下文件夹名）
        "kwargs": {},         # 模型专属超参，例如 hidden_dim / num_layers
    },

    # ---- 损失 ----
    "loss": {
        "name":   None,       # 损失名（对应 models/<name>/loss.py 里注册的名字）
        "kwargs": {},         # 损失专属超参，例如 class_weight / label_smoothing
    },

    # ---- 数据 ----
    "data": {
        "name":  "cora",      # 数据集名（对应 src/data/loader.py 里注册的名字）
        "root":  "data",      # 数据存放根目录
        "split": "public",    # 划分方式："public" / "random" / "train_rest" / "test_rest"
    },

    # ---- 优化器 ----
    "optim": {
        "lr":           1e-2,
        "weight_decay": 5e-4,
        "scheduler":    None,   # None / "cosine" / "step"
    },

    # ---- 训练流程 ----
    "train": {
        "epochs":     200,
        "seed":       42,
        "device":     "auto",   # auto / cuda / mps / cpu
        "grad_clip":  None,     # 梯度裁剪阈值，None 表示不裁剪
        "early_stop": 100,      # 早停耐心值，None 表示不启用
        "log_every":  1,        # 每多少轮打印一次日志
    },

    # ---- 保存策略 ----
    "save": {
        "monitor": "val_acc",   # 用哪个指标判断"最佳"
        "mode":    "max",       # max 越大越好，min 越小越好
    },

    # ---- 日志 ----
    "log": {
        "level":   "INFO",
        "backend": "csv",       # csv / tensorboard / both（当前实现用 csv）
    },

    # ---- 实验目录 ----
    "exp": {
        "name":       None,     # 实验名，None 时自动用模型名
        "output_dir": "runs",   # 输出根目录
    },
}