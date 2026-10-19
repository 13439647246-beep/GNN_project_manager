# ============================================================
# src/engine/runner.py
# 训练编排：把"一次完整训练"和"遍历多个 arg 组合"统一封装。
#
# 职责划分：
#   - trainer.py : 训练一个 epoch（Trainer 类）
#   - runner.py  : 编排一次 / 多次训练（train_once / run_grid）
#   - config/grid.py : 只负责展开参数组合，不做训练
# ============================================================

import torch
# ============================================================
# src/engine/runner.py
# 训练编排：一次完整训练 + 遍历多个 arg 组合。
# ============================================================


from pathlib import Path

from src.config import expand_grid
from src.utils.registry import build_model, build_loss
from src.engine.logger import Logger
from src.engine.checkpoint import Checkpoint
from src.engine.trainer import Trainer
from src.engine.run_layout import (
    finalize_best, grid_run_dir, hyperparams_str,
    make_timestamp, single_run_dir, write_summary,
)


# ------------------------------------------------------------
# 单次训练：给定一份 arg 和数据，跑完 训练 + 测试
# 返回值兼容原版字段：exp_name / best_val / test_acc / test_loss
# ------------------------------------------------------------
def train_once(arg, data, run_dir=None, source_mode="single",
               combo=None, hyperparams=None, exp_name=None):
    # ★ 保持你原版：Logger 若不支持 run_dir，先改 Logger 再传
    logger = Logger(arg, run_dir=run_dir) if run_dir is not None else Logger(arg)
    try:
        ckpt = Checkpoint(arg, logger)
        model = build_model(arg)   # ★ 补 import 或保持原版
        criterion = build_loss(arg)

        # ★ 保持你原版 optimizer 构建，下面是示例
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=arg["optim"]["lr"],
            weight_decay=arg["optim"].get("weight_decay", 0.0),
        )

        if hasattr(model, "prepare_input"):
            model.prepare_input(data)

        trainer = Trainer(arg, model, data, criterion, optimizer, ckpt, logger)
        trainer.fit()

        test_metrics = {}
        if ckpt.best_path.exists():
            test_metrics = trainer.test(str(ckpt.best_path)) or {}

        with torch.no_grad():
            out = model(data)
            C, mu, beta = out["C"], out["mu"], out["beta"]

        return {
            # 兼容原版 print_summary / save_summary
            "exp_name":    exp_name or combo or arg.get("exp", "single"),
            "best_val":    ckpt.best_value,
            "test_acc":    test_metrics.get("test_acc"),
            "test_loss":   test_metrics.get("test_loss"),
            "run_dir":     str(logger.run_dir),
            "error":       None,
            # 新增：供 _best / summary 使用
            "combo":       combo,
            "hyperparams": hyperparams,
            "best_value":  ckpt.best_value,
            "best_epoch":  ckpt.best_epoch,
            "best_path":   str(ckpt.best_path),
            "monitor":     ckpt.monitor,
            "save_mode":   ckpt.mode,
            "source_mode": source_mode,
            "model_name":  arg["model"]["name"],
            "dataset":     arg["data"]["name"],
            "arg":         arg,
        }
    finally:
        logger.close()


def run_single(arg, data, ts=None):
    model_name = arg["model"]["name"]
    dataset = arg["data"]["name"]
    model_root = Path(arg["exp"]["output_dir"]) / model_name
    ts = ts or make_timestamp(model_root)          # ★ 外部传入优先
    ts_root = model_root / ts
    exp = arg["exp"].get("name") or "single"
    run_dir = single_run_dir(model_root, ts, dataset, exp)

    result = train_once(arg, data, run_dir=run_dir, source_mode="single", exp_name=exp)
    finalize_best(ts_root, dataset, [result], arg)
    return result


def run_grid(base_arg, grid, data, ts=None):
    model_name = base_arg["model"]["name"]
    dataset = base_arg["data"]["name"]
    model_root = Path(base_arg["exp"]["output_dir"]) / model_name
    ts = ts or make_timestamp(model_root)          # ★ 外部传入优先
    ts_root = model_root / ts
    keys = sorted(grid.keys())

    results = []
    for i, arg in enumerate(expand_grid(base_arg, grid), start=1):
        combo = f"combo{i}"
        run_dir = grid_run_dir(model_root, ts, dataset, combo)
        hp = hyperparams_str(arg, keys)
        print(f"[grid] {combo} {hp}")
        r = train_once(
            arg, data, run_dir=run_dir, source_mode="grid",
            combo=combo, hyperparams=hp, exp_name=combo,
        )
        results.append(r)
        print(f"[grid] {combo} best_val={r.get('best_val')} monitor={r.get('monitor')}")

    save_summary(results, base_arg, ts_root / dataset / "_grid" / "summary.csv")
    finalize_best(ts_root, dataset, results, base_arg)
    print_summary(results, model_name, dataset)
    return results


# ============================================================
# 汇总：控制台表格（保持你原版风格）
# ============================================================
def print_summary(results, model_name, data_name):
    print("\n" + "=" * 96)
    print(f"[grid] 汇总  model={model_name}  dataset={data_name}")
    print("=" * 96)
    header = f"{'exp_name':<46s}{'model':>8s}{'dataset':>10s}{'best_val':>11s}{'test_acc':>11s}"
    print(header)
    print("-" * 96)
    for r in results:
        bv = f"{r['best_val']:.4f}" if r.get("best_val") is not None else "-"
        ta = f"{r['test_acc']:.4f}" if r.get("test_acc") is not None else "-"
        print(f"{r['exp_name']:<46s}{model_name:>8s}{data_name:>10s}{bv:>11s}{ta:>11s}")
    print("=" * 96)

    valid = [r for r in results if r.get("best_val") is not None]
    if valid:
        mode = results[0].get("save_mode", "max")
        best = (max(valid, key=lambda x: x["best_val"]) if mode == "max"
                else min(valid, key=lambda x: x["best_val"]))
        print(
            f"[grid] 最佳组合：{best['exp_name']}  "
            f"model={model_name}  dataset={data_name}  "
            f"{best.get('monitor', 'best_val')} = {best['best_val']:.4f}\n"
        )


# ============================================================
# 汇总：写 CSV（只保留一个实现，路径在 timestamp 下）
# ============================================================
def save_summary(results, base_arg, out_path=None):
    """写 <ts_root>/<dataset>/_grid/summary.csv。"""
    if out_path is None:
        # 兼容旧调用：落到 <output_dir>/<model>/<dataset>/_grid/summary.csv
        out_path = (Path(base_arg["exp"]["output_dir"]) / base_arg["model"]["name"]
                    / base_arg["data"]["name"] / "_grid" / "summary.csv")
    write_summary(results, Path(out_path))
    print(f"[grid] 汇总已保存：{out_path}\n")