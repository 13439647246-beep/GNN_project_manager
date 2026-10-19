# ============================================================
# src/config/__init__.py
# 超参管理对外接口。
# 调用方只需 from src.config import load_arg 即可。
# ============================================================

from src.config.default_arg import DEFAULT_ARG
from src.config.arg_io import (
    load_arg,
    validate_arg,
    load_yaml,
    save_yaml,
    get_nested,
    set_nested,
    deep_merge,
    ensure_model_yaml,
)
from src.config.grid import expand_grid

__all__ = [
    "DEFAULT_ARG",
    "load_arg", "validate_arg",
    "load_yaml", "save_yaml",
    "get_nested", "set_nested", "deep_merge",
    "ensure_model_yaml",
    "expand_grid",
]
