# ============================================================
# run_test.py
# ============================================================
from pathlib import Path

from src.config import load_arg, validate_arg, ensure_model_yaml
from src.utils.registry import list_models, list_losses
from src.data import prepare_data, describe_data
from src.engine.runner import run_single, run_grid
from src.engine.run_layout import make_timestamp


import models  # 触发模型 / 损失注册

# ============================================================
# 配置区：只改这里
# ============================================================
MODEL_NAME = "HGCNRH"
DATA_NAMES = ["cora", "citeseer", "pubmed", ]   # ★ 改成列表
MODE = "grid"

# HGCNRH 超参数
GRID = {
    "optim.lr": [0.001, 0.01, 0.05],
    "optim.weight_decay": [0.01, 0.001, 0.0001],
    "model.kwargs.hidden_dim": [128, 256],
    "model.kwargs.qk_dim": [64, 128],
    "model.kwargs.dropout": [0.5, 0.3, 0.1],
    "model.kwargs.kmeans_n": [10, 20, 30],
    "loss.kwargs.lambda_kl": [0.05, 0.1, 0.5, 1.0],
}


# ============================================================

def main():
    # 1. 先用第一个数据集拿 output_dir，生成本次运行统一的 timestamp
    first_cfg = ensure_model_yaml(MODEL_NAME, data_name=DATA_NAMES[0])
    first_arg = load_arg(first_cfg, overrides={
        "model.name": MODEL_NAME,
        "data.name":  DATA_NAMES[0],
    })
    validate_arg(first_arg)
    model_root = Path(first_arg["exp"]["output_dir"]) / MODEL_NAME
    ts = make_timestamp(model_root)

    # 2. 逐数据集：重新 load_arg + prepare_data，共享同一个 ts
    for data_name in DATA_NAMES:
        cfg_path = ensure_model_yaml(MODEL_NAME, data_name=data_name)
        arg = load_arg(cfg_path, overrides={
            "model.name": MODEL_NAME,
            "data.name":  data_name,
        })
        validate_arg(arg)

        print(f"\n[arg] model={arg['model']['name']}  "
              f"loss={arg['loss']['name']}  data={arg['data']['name']}  ts={ts}")

        data = prepare_data(arg["data"]["name"], arg)
        describe_data(data, arg["data"]["name"])

        if MODE == "single":
            run_single(arg, data, ts=ts)
        elif MODE == "grid":
            run_grid(arg, GRID, data, ts=ts)
        else:
            raise ValueError(f"未知 MODE: {MODE}")


if __name__ == "__main__":
    main()