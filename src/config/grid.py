# ============================================================
# src/config/grid.py
# 网格搜索：把参数列表展开成所有组合，每个组合产出一份完整 arg。
#
# 与命令行无关，直接在 run_test.py 里以 dict 的形式给出网格：
#   GRID = {
#       "optim.lr":              [0.1, 0.01, 0.001],
#       "model.kwargs.dropout":  [0.0, 0.5],
#   }
# ============================================================

import copy
import itertools

from src.config.arg_io import set_nested


def expand_grid(base_arg, grid):
    """
    网格展开生成器。

    参数:
        base_arg : 已加载好的完整 arg（作为所有组合的基底）
        grid     : {"点号路径": [候选值, ...]}

    每次产出:
        一份 base_arg 的深拷贝，其中点号路径上的值被替换为本次组合的值，
        并且 exp.name 会被改写为可读的实验名（用于 runs/ 目录命名）。
    """
    # 空网格：只产出基底一份，方便调用方统一处理
    if not grid:
        yield _with_exp_name(copy.deepcopy(base_arg), applied={})
        return

    keys = list(grid.keys())
    # 允许值是标量（非列表），统一成列表再 product
    values = [list(v) if isinstance(v, (list, tuple)) else [v]
              for v in grid.values()]

    # itertools.product 做笛卡尔积，key 顺序与 values 对齐
    for combo in itertools.product(*values):
        arg = copy.deepcopy(base_arg)
        applied = {}
        for k, v in zip(keys, combo):
            set_nested(arg, k, v)       # 点号路径深层赋值
            applied[k] = v
        yield _with_exp_name(arg, applied)


def _with_exp_name(arg, applied):
    """
    根据本次网格组合生成实验名。
    例如：exp.name = "gcn__lr=0.01__dropout=0.5"
    这样 runs/<model>/<exp>/<时间戳>/ 目录一眼能看出跑的是哪组参数。
    """
    base = arg["exp"].get("name") or arg["model"]["name"] or "exp"
    if not applied:
        arg["exp"]["name"] = base
        return arg
    # 只取路径末段，避免实验名过长
    parts = [f"{k.split('.')[-1]}={v}" for k, v in applied.items()]
    arg["exp"]["name"] = f"{base}__" + "__".join(parts)
    return arg