# ============================================================
# src/layers/attention.py
# 通用注意力层。
# ============================================================
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.basic import get_activation   # ★ 复用统一的激活工厂


class MultiHeadSelfAttention(nn.Module):
    """
    标准多头自注意力（无 mask 版本）。

    实现要点：
      - 用一个 Linear 一次性算出 Q / K / V（在特征维度上切三份）
      - reshape 成多头后做 scaled dot-product attention
      - 输出投影后可选加激活

    参数:
        dim         : 输入 / 输出维度
        num_heads   : 头数，必须整除 dim
        dropout     : 注意力权重 dropout
        bias        : QKV / 输出投影是否带偏置
        activation  : 激活名（"relu" / "gelu" / ...）；
                      None 表示跳过激活（用 nn.Identity）
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 dropout=0.0,
                 bias=True,
                 activation=None):            # ★ 新增参数，默认不加激活
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # QKV 一次算完，输出投影单独一层
        self.qkv  = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout  = nn.Dropout(dropout)

        # ★ 激活：None → Identity，直接跳过；否则用 get_activation 取模块
        self.activation = get_activation(activation)

    def forward(self, x):
        """
        参数:
            x : (N, dim) 的节点特征矩阵
        返回:
            (N, dim) 的注意力输出
        """
        N = x.size(0)

        # QKV: (N, 3*dim) → 拆成 3 份 → 每份 reshape 成 (N, H, D)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def _split(t):
            return t.view(N, self.num_heads, self.head_dim).transpose(0, 1)
            # → (H, N, D)

        q, k, v = _split(q), _split(k), _split(v)

        # scaled dot-product attention: (H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和 → (H, N, D) → (N, H, D) → (N, dim)
        out = (attn @ v).transpose(0, 1).contiguous().view(N, self.dim)
        out = self.proj(out)

        # ★ 激活可选：activation 是 Identity 时等价于原样返回
        out = self.activation(out)
        return self.out_dropout(out)



class BiaffineAttention(nn.Module):
    """
    双仿射注意力层
    score(i, j) = h_i^T U g_j + W1 h_i + W2 g_j + b
    alpha = softmax_j(score)
    output_i = sum_j alpha_ij * v_j

    参数
    query_dim : int
        query 表示维度 d_q
    key_dim : int
        key/value 表示维度 d_k
    out_dim : int
        输出维度（value 投影后维度）
    bias : bool
        是否使用 W1, W2, b
    """

    def __init__(self,
                 query_dim,
                 key_dim,
                 out_dim=None,
                 bias=False,
                 activation="softmax"):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.out_dim = out_dim if out_dim is not None else key_dim
        self.use_bias = bias
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.scale = 1.0 / math.sqrt(query_dim)      # ★ 新增

        # 双线性权重 U: (d_q, d_k)
        self.U = nn.Parameter(torch.empty(query_dim, key_dim))

        # 线性投影 W1: (d_q,), W2: (d_k,)
        if bias:
            self.W1 = nn.Parameter(torch.empty(query_dim))
            self.W2 = nn.Parameter(torch.empty(key_dim))
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('W1', None)
            self.register_parameter('W2', None)
            self.register_parameter('b', None)

        # value 投影（若 out_dim != key_dim 时需要）
        self.V_proj = nn.Linear(key_dim, self.out_dim, bias=False) \
            if self.out_dim != key_dim else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        if self.use_bias:
            nn.init.xavier_uniform_(self.W1.unsqueeze(0))
            nn.init.xavier_uniform_(self.W2.unsqueeze(0))
            nn.init.zeros_(self.b)

    def forward(self, query, key, value=None, mask=None):
        squeeze = False
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            if value is not None:
                value = value.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze = True

        if value is None:
            value = key

        # 双线性项 + scale
        q_proj = torch.matmul(query, self.U)
        bilinear = torch.matmul(q_proj, key.transpose(1, 2)) * self.scale  # ★ scale

        if self.use_bias:
            linear_q = torch.einsum('bnd,d->bn', query, self.W1).unsqueeze(-1)
            linear_k = torch.einsum('bmd,d->bm', key, self.W2).unsqueeze(1)
            scores = bilinear + linear_q + linear_k + self.b
        else:
            scores = bilinear

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # ★ 强制 softmax：双仿射注意力的定义就是 softmax
        if self.activation_name in (None, "none", "identity"):
            attn = torch.softmax(scores, dim=-1)
        else:
            attn = self.activation(scores)

        v = self.V_proj(value)
        output = torch.matmul(attn, v)

        if squeeze:
            output = output.squeeze(0)
            attn = attn.squeeze(0)
        return output, attn