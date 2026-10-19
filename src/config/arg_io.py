# ============================================================
# src/config/arg_io.py
# arg 的读取 / 保存 / 深合并 / 点号路径覆盖 / 校验。
#
# 这是"配置驱动"的基础工具，与命令行无关。
# 加载顺序（后者覆盖前者）：
#   ① DEFAULT_ARG（代码兜底）
#   ② configs/_base.yaml（全局公共）
#   ③ 传入的 cfg_path（模型专属 yaml）
#   ④ overrides（点号路径，来自脚本或网格搜索）
# ============================================================

import copy
import os
import yaml

from src.config.default_arg import DEFAULT_ARG


# ------------------------------------------------------------
# 点号路径的读写：支持 "optim.lr" / "model.kwargs.dropout" 这种深层访问
# ------------------------------------------------------------
def get_nested(d, path, default=None):
    """按 'a.b.c' 路径读取 d 里的值，路径不存在返回 default"""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested(d, path, value):
    """
    按 'a.b.c' 路径写入 d。
    中间缺失的层级自动创建为 dict，方便网格搜索覆盖深层参数。
    """
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


# ------------------------------------------------------------
# 深合并：递归合并两个 dict
# ------------------------------------------------------------
def deep_merge(base, override):
    """
    递归合并两个 dict：
      - 双方都是 dict → 递归合并
      - 否则 override 直接覆盖 base
      - 全程 deepcopy，不修改任何入参
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ------------------------------------------------------------
# yaml 读写
# ------------------------------------------------------------
def load_yaml(path):
    """安全读取 yaml；路径为 None 或不存在时返回空 dict"""
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj, path):
    """把 arg 快照写成 yaml（用于 runs/ 目录下的可追溯记录）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


# ------------------------------------------------------------
# 主入口：加载完整 arg
# ------------------------------------------------------------
def load_arg(cfg_path, overrides=None, base_path="configs/_base.yaml"):
    """
    加载完整 arg。

    参数:
        cfg_path   : 模型专属 yaml 路径（可传 None）
        overrides  : dict，点号路径覆盖，如 {"optim.lr": 0.001}
        base_path  : 全局 yaml 路径，None 表示跳过

    返回:
        完整的 arg dict
    """
    arg = copy.deepcopy(DEFAULT_ARG)
    arg = deep_merge(arg, load_yaml(base_path))     # ② 全局公共
    arg = deep_merge(arg, load_yaml(cfg_path))      # ③ 模型专属

    if overrides:                                    # ④ 点号路径覆盖
        for k, v in overrides.items():
            set_nested(arg, k, v)

    # exp.name 兜底：没写就用模型名，方便 runs/ 目录命名
    if not arg["exp"]["name"]:
        arg["exp"]["name"] = arg["model"]["name"] or "exp"

    return arg


# ------------------------------------------------------------
# 校验：防止 grid 里写错 key 导致静默失败
# ------------------------------------------------------------
def validate_arg(arg, require=("model.name", "loss.name", "data.name")):
    """
    检查关键字段是否齐全。

    参数:
        arg     : 完整 arg dict
        require : 必须非空的点号路径列表

    失败时抛 ValueError，并列出所有问题（一次性报全，不是遇到就停）。
    """
    errors = []
    for path in require:
        v = get_nested(arg, path)
        if v is None or v == "":
            errors.append(f"  - '{path}' 未设置")
    if errors:
        raise ValueError("arg 缺少必填字段：\n" + "\n".join(errors))


# ============================================================
# 确保某个模型的 yaml 存在：不存在则基于 _base.yaml 生成骨架。
# ============================================================
def ensure_model_yaml(model_name, data_name=None, config_dir="configs"):
    """
    确保 configs/<model_name>.yaml 存在，返回其路径。

    行为：
      - 已存在 → 直接返回路径，不修改原文件
      - 不存在 → 以 _base.yaml 为骨架生成一份，含：
            model.name = <model_name>
            loss.name  = <model_name>_loss
            data.name  = <data_name>（未传时用 "cora"）
        _base.yaml 里已配置的字段（train / optim / log 等）原样拷贝。

    参数:
        model_name : 模型名，如 "gcn"
        data_name  : 数据集名，如 "cora"；None 时用 "cora"
        config_dir : 配置目录，默认 "configs"

    返回:
        yaml 文件的完整路径
    """
    path = os.path.join(config_dir, f"{model_name}.yaml")
    if os.path.exists(path):
        return path

    # 读 _base.yaml 作为骨架；不存在时用空 dict
    base = load_yaml(os.path.join(config_dir, "_base.yaml"))

    # 组装完整骨架：先放 base，再叠加 model / loss / data 三块
    cfg = dict(base)
    cfg["model"] = {"name": model_name, "kwargs": {}}
    cfg["loss"]  = {"name": f"{model_name}_loss", "kwargs": {}}
    cfg["data"]  = {"name": data_name or "cora", "split": "public"}

    save_yaml(cfg, path)
    print(f"[config] 未找到 {path}，已基于 _base.yaml 生成骨架")
    return path