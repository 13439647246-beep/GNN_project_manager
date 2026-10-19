import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GCN, MLP, BiaffineAttention, MultiHeadSelfAttention


class GCNRHLayer(nn.Module):
    def __init__(self,
                 in_dim,  # 输入维度：n+δ
                 hidden_dim,  # mlp维度：d
                 qk_dim,  # 双仿射注意力q维度：d
                 out_dim,  # 输出维度
                 gcn_layers=1,
                 mlp_layers=1,
                 dropout=0.5):
        super().__init__()

        self.GCN = GCN(in_dim, hidden_dim, qk_dim,
                       num_layers=gcn_layers, norm=None, activation="tanh",
                       dropout=dropout, residual=False)
        self.MLP = MLP(in_dim, hidden_dim, qk_dim,
                       num_layers=mlp_layers, activation="tanh", dropout=dropout)

        self.BiaffineAttentionH_d = BiaffineAttention(qk_dim, qk_dim, out_dim=out_dim, bias=False)
        self.BiaffineAttentionH_ego = BiaffineAttention(qk_dim, qk_dim, out_dim=out_dim, bias=False)

        self.proj_d = nn.Linear(qk_dim, out_dim)
        self.proj_ego = nn.Linear(qk_dim, out_dim)

        self.norm_mu = nn.BatchNorm1d(out_dim)
        self.norm_beta = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        H_d = self.GCN(x, edge_index)
        H_ego = self.MLP(x)

        # H_d, H_ego：n * d
        # Hp_d, Hp_ego：n * d
        Hp_d, attn_d = self.BiaffineAttentionH_d(H_d, H_ego)
        Hp_ego, attn_ego = self.BiaffineAttentionH_ego(H_ego, H_d)

        mu = self.norm_mu(self.proj_d(H_d) + Hp_d)
        log_var = self.norm_beta(self.proj_ego(H_ego) + Hp_ego)
        beta = F.softplus(log_var) + 1e-6  # 保证为正，不用 exp 防溢出
        eps = torch.randn_like(beta)
        C = mu + eps * beta
        # C, mu, beta：n * n
        return C, mu, beta
