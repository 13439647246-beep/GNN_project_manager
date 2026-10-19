# ============================================================
# src/layers/initialization.py
# 权重初始化策略。
#
# 用法:
#   apply_init(model, scheme="xavier")            # 全模型
#   apply_init(model, scheme="kaiming", mode="fan_out")
#
# 说明：
#   - 只对 nn.Linear / nn.Conv* 生效
#   - 偏置统一初始化为 0
#   - 归一化层的 weight / bias 通常用默认，不做处理
# ============================================================

import torch.nn as nn


# ------------------------------------------------------------
# 支持的初始化方式
# ------------------------------------------------------------
def xavier_init(module, gain=1.0):
    """Xavier / Glorot 初始化，适合 tanh / sigmoid 类激活"""
    nn.init.xavier_uniform_(module.weight, gain=gain)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def kaiming_init(module, mode="fan_in", nonlinearity="relu"):
    """Kaiming / He 初始化，适合 ReLU 类激活"""
    nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def orthogonal_init(module, gain=1.0):
    """正交初始化，常用于 RNN 与需要保持能量的场景"""
    nn.init.orthogonal_(module.weight, gain=gain)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def normal_init(module, mean=0.0, std=0.01):
    """正态初始化（标准差很小），适合浅层网络或 embedding"""
    nn.init.normal_(module.weight, mean=mean, std=std)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


# 名字 → 函数 的映射，让配置里可以用字符串指定
_INIT_SCHEMES = {
    "xavier":     xavier_init,
    "kaiming":    kaiming_init,
    "orthogonal": orthogonal_init,
    "normal":     normal_init,
}


# ------------------------------------------------------------
# 批量应用
# ------------------------------------------------------------
def apply_init(model, scheme="xavier", **kwargs):
    """
    对整个模型应用指定的初始化。

    参数:
        model  : nn.Module
        scheme : "xavier" / "kaiming" / "orthogonal" / "normal"
        kwargs : 传给具体初始化函数的参数（如 gain / mode / std）

    返回:
        model（原地修改并返回，便于链式调用）
    """
    if scheme not in _INIT_SCHEMES:
        raise ValueError(
            f"未知初始化方案 '{scheme}'，可选：{sorted(_INIT_SCHEMES)}"
        )
    fn = _INIT_SCHEMES[scheme]

    def _apply(m):
        # 只对带 weight 的 Linear / Conv 生效
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fn(m, **kwargs)

    model.apply(_apply)
    return model