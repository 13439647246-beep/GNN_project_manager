# src/engine/run_layout.py
"""运行目录布局、网格命名、最优模型汇总。"""
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def runs_root(arg):
    if "exp" in arg and "output_dir" in arg["exp"]:
        return Path(arg["exp"]["output_dir"])
    return Path(arg.get("run", {}).get("root", "runs"))


def make_timestamp(model_root: Path) -> str:
    """同一秒内多次运行加 _01 后缀。"""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not (model_root / ts).exists():
        return ts
    for i in range(1, 100):
        cand = f"{ts}_{i:02d}"
        if not (model_root / cand).exists():
            return cand
    raise RuntimeError(f"timestamp collision under {model_root}")


def single_run_dir(model_root, timestamp, dataset, exp):
    return model_root / timestamp / dataset / "_single" / exp


def grid_run_dir(model_root, timestamp, dataset, combo):
    return model_root / timestamp / dataset / "_grid" / combo


def best_dir(ts_root, dataset):
    return ts_root / "_best" / dataset


def get_nested(d, path):
    cur = d
    for k in path.split("."):
        cur = cur[k]
    return cur


def hyperparams_str(arg, keys):
    """按网格 key 生成可读摘要，如 lr=0.1,dropout=0.5。"""
    parts = []
    for k in sorted(keys):
        v = get_nested(arg, k)
        parts.append(f"{k.split('.')[-1]}={v}")
    return ",".join(parts)


def select_best(results):
    """grid 优先，single 兜底；按 save_mode 选 max/min。"""
    grid = [r for r in results if r.get("source_mode") == "grid" and r.get("best_value") is not None]
    pool = grid if grid else [r for r in results if r.get("best_value") is not None]
    if not pool:
        return None
    mode = pool[0].get("save_mode", "max")
    key = lambda r: r["best_value"]
    return max(pool, key=key) if mode == "max" else min(pool, key=key)


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def finalize_best(ts_root: Path, dataset: str, results, base_arg):
    """把本次运行最优模型汇总到 _best/<dataset>/。"""
    best = select_best(results)
    if best is None:
        print(f"[_best] no valid best for {dataset}, skip (check save.monitor)")
        return None

    out = best_dir(ts_root, dataset)
    out.mkdir(parents=True, exist_ok=True)

    # config.yaml：优先用 result 里的完整 arg
    arg = best.get("arg")
    if arg is not None:
        with open(out / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(arg, f, allow_unicode=True, sort_keys=False)
    else:
        src_cfg = Path(best["run_dir"]) / "config.yaml"
        if src_cfg.exists():
            shutil.copy2(src_cfg, out / "config.yaml")

    # best.pt：软链到来源，失败则复制
    src_best = Path(best["best_path"])
    if src_best.exists():
        _link_or_copy(src_best, out / "best.pt")

    metrics = {
        "model": best.get("model_name"),
        "dataset": dataset,
        "timestamp": ts_root.name,
        "source_mode": best.get("source_mode"),
        "combo": best.get("combo"),
        "hyperparams": best.get("hyperparams"),
        "monitor": best.get("monitor"),
        "save_mode": best.get("save_mode"),
        "best_value": best.get("best_value"),
        "best_epoch": best.get("best_epoch"),
        "source_dir": best.get("run_dir"),
        "best_path": str(out / "best.pt"),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    readme = (
        f"# Best model\n\n"
        f"- dataset: `{dataset}`\n"
        f"- source: `{best.get('source_mode')}` / `{best.get('combo')}`\n"
        f"- hyperparams: `{best.get('hyperparams')}`\n"
        f"- monitor: `{best.get('monitor')}` = `{best.get('best_value')}` (epoch {best.get('best_epoch')})\n"
        f"- source dir: `{best.get('run_dir')}`\n\n"
        f"Load:\n\n```python\nfrom src.engine.checkpoint import load_best_model\n"
        f"model, arg, meta = load_best_model('{out / 'best.pt'}')\n```\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")

    # index.json
    idx_path = ts_root / "_best" / "index.json"
    idx = {}
    if idx_path.exists():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    idx.setdefault("model", best.get("model_name"))
    idx.setdefault("timestamp", ts_root.name)
    idx.setdefault("datasets", {})[dataset] = {
        "best_path": str(out / "best.pt"),
        "source_dir": best.get("run_dir"),
        "source_mode": best.get("source_mode"),
        "combo": best.get("combo"),
        "monitor": best.get("monitor"),
        "best_value": best.get("best_value"),
    }
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[_best] {dataset} -> {out} (source={best.get('combo') or best.get('source_mode')})")
    return out


def write_summary(results, out_path: Path):
    """按 monitor 排序写 summary.csv。"""
    if not results:
        return
    mode = results[0].get("save_mode", "max")
    rows = sorted(results, key=lambda r: (r.get("best_value") is None, r.get("best_value") or 0),
                  reverse=(mode == "max"))
    cols = ["combo", "hyperparams", "best_value", "monitor", "best_epoch",
            "best_path", "run_dir", "source_mode"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})