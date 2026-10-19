# ============================================================
# src/layers/basic.py
# 通用基础层与工具：激活函数、MLP、残差块。
#
# 这些层不依赖图结构，任何模型（图/非图）都能复用。
# ============================================================

import torch
import torch.nn as nn
import functools


# ------------------------------------------------------------
# 激活函数工厂：让模型配置用字符串指定激活
# ------------------------------------------------------------
_ACTIVATIONS = {
    "relu":     nn.ReLU,
    "relu6":    nn.ReLU6,
    "leaky":    nn.LeakyReLU,
    "elu":      nn.ELU,
    "gelu":     nn.GELU,
    "silu":     nn.SiLU,       # 又叫 Swish
    "swish":    nn.SiLU,
    "tanh":     nn.Tanh,
    "sigmoid":  nn.Sigmoid,
    "identity": nn.Identity,
    "softmax":  functools.partial(nn.Softmax, dim=-1),   # ★ 必须带 dim
}

def get_activation(name):
    """
    按名字取激活函数模块。name 为 None 时返回 Identity。

    用法:
        act = get_activation("relu")
        act = get_activation("gelu")
    """
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"unknown activation: {name}")
    return _ACTIVATIONS[name]()


# ------------------------------------------------------------
# MLP：多层感知机
# ------------------------------------------------------------
class MLP(nn.Module):
    """
    多层感知机：Linear → Act → Dropout 重复 num_layers-1 次，
    最后一层只做 Linear，不带激活/ dropout。

    参数:
        in_dim     : 输入维度
        hidden_dim : 隐层维度
        out_dim    : 输出维度
        num_layers : 总层数（>=1）
        activation : 激活名（"relu" / "gelu" / ...）
        dropout    : dropout 概率（0 表示不用）
        bias       : Linear 是否带偏置
        last_bias  : 输出层是否带偏置（默认跟 bias）
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_layers=2, activation="relu", dropout=0.0,
                 bias=True, last_bias=None):
        super().__init__()
        assert num_layers >= 1, "num_layers 至少为 1"

        if last_bias is None:
            last_bias = bias

        act = get_activation(activation)
        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim, bias=last_bias))
        else:
            # 首层：in → hidden
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # 中间层：hidden → hidden
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # 末层：hidden → out，不带激活
            layers.append(nn.Linear(hidden_dim, out_dim, bias=last_bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# 残差块：y = x + f(x)
# ------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    通用残差块：先过一个子模块 f，再与输入相加。

    支持两种用法：
        - in_dim == out_dim：直接相加
        - in_dim != out_dim：用线性投影把输入投到 out_dim 再相加

    参数:
        block     : 主分支子模块（nn.Module），如 MLP
        in_dim    : 输入维度
        out_dim   : 输出维度
        activation: 相加后是否激活（None 表示不激活）
    """
    def __init__(self, block, in_dim, out_dim,
                 activation="relu"):
        super().__init__()
        self.block = block
        # 维度不一致时，用 1x1 线性投影对齐
        self.proj = (nn.Identity() if in_dim == out_dim
                     else nn.Linear(in_dim, out_dim, bias=False))
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.proj(x) + self.block(x))