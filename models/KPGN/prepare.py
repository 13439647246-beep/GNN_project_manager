import torch.nn as nn
import torch.nn.init as init
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils import to_dense_adj
from torch.nn import functional
import os
import torch
import networkx as nx
import scipy.sparse as sp
import copy
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm

# 计算对称归一化邻接矩阵 A_hat = D^{-1/2} (A + I) D^{-1/2}
def normalize_adj(adj, add_self_loops=True):
    N = adj.size(0)

    # ---- 加自环 ----
    if add_self_loops:
        index = torch.arange(N, device=adj.device)
        eye_indices = torch.stack([index, index], dim=0)
        eye_values = torch.ones(N, device=adj.device, dtype=adj.values().dtype)
        I = torch.sparse_coo_tensor(
            eye_indices, eye_values, (N, N), device=adj.device
        ).coalesce()
        adj = (adj + I).coalesce()

    # ---- 计算度 D ----
    indices = adj.indices()
    values = adj.values()
    deg = torch.zeros(N, device=adj.device, dtype=values.dtype)
    deg.index_add_(0, indices[0], values)

    # ---- D^{-1/2}，孤立点置 0 ----
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # ---- 对称归一化 ----
    row, col = adj.indices()
    norm_values = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]

    adj_hat = torch.sparse_coo_tensor(
        adj.indices(), norm_values, adj.size(), device=adj.device
    ).coalesce()

    return adj_hat


def build_adjacency_dict_sparse(adj, remove_self_loop=True):
    adj = adj.coalesce()
    N = adj.size(0)
    adjacency_dict = {i: [] for i in range(N)}

    indices = adj.indices()   # shape [2, nnz]
    values = adj.values()     # shape [nnz]

    rows, cols = indices[0], indices[1]

    for i, j, v in zip(rows.tolist(), cols.tolist(), values.tolist()):
        if v == 0:
            continue
        if remove_self_loop and i == j:
            continue
        adjacency_dict[i].append(j)

    return adjacency_dict


# 利用相似度采样
def similary_sampling(adjacency_dict, nodes_num, getnei_num=20):
    nei_adj_dict = {}
    for i in range(0,nodes_num):
        nei_adj_dict.setdefault(i,[])
        nei_i = adjacency_dict[i]
        node_i_degree = len(nei_i)
        if node_i_degree<= getnei_num:
            for k in nei_i:
                nei_adj_dict.setdefault(i,[]).append(k)

        else:
            i_nei_list = []
            for j in adjacency_dict[i]:
                j_sim_list = [j]
                nei_j = adjacency_dict[j]
                node_j_degree = len(nei_j)
                # 寻找交集
                similen_ij = len(list(set(nei_i).intersection(nei_j)))
                similary_ij = (node_i_degree-1)*(node_j_degree-1)/((node_j_degree+node_i_degree-2-similen_ij)**2)
                j_sim_list.append(similary_ij)
                i_nei_list.append(j_sim_list)
                # return
            i_nei_list.sort(reverse=True,key=lambda x: x[1])
            for k in range(0,getnei_num):
                sele_neik = i_nei_list[k][0]
                nei_adj_dict.setdefault(i,[]).append(sele_neik)
    return nei_adj_dict



def torch_adj_to_nx(adj, remove_self_loop=True):
    if adj.is_sparse:
        adj = adj.coalesce()
        idx = adj.indices().cpu().numpy()
        val = adj.values().cpu().numpy()
        rows, cols = idx[0], idx[1]
    else:
        coo = sp.coo_matrix(adj.cpu().numpy())
        rows, cols, val = coo.row, coo.col, coo.data

    G = nx.Graph()
    G.add_nodes_from(range(adj.size(0)))
    for i, j, v in zip(rows, cols, val):
        if v == 0:
            continue
        if remove_self_loop and i == j:
            continue
        G.add_edge(int(i), int(j), weight=float(v))
    return G

# =========================================================
# BS 算法：在 G 中寻找包含约束集 F 的最大 k-plex
# =========================================================
def bs_max_kplex(G, k, F, bound, time_limit=None):
    """
    用 Branch-and-Search 在 G 中寻找包含约束集 F 的最大 k-plex。
    使用显式栈的迭代版本，避免 RecursionError。

    参数：
        G         : networkx.Graph
        k         : k-plex 参数
        F         : 约束节点列表
        bound     : k-plex 最小节点数
        time_limit: 可选，超时秒数

    返回：
        节点列表，找不到满足 bound 的返回 None
    """
    import time as _time

    start_time = _time.time()
    F_set = frozenset(F)
    n_F = len(F_set)

    # 预计算邻接集，加速 has_edge
    adj = {u: set(G.neighbors(u)) for u in G.nodes()}

    def is_kplex(S):
        n = len(S)
        for u in S:
            deg_in = len(adj[u] & S)
            if deg_in < n - k:
                return False
        return True

    if not is_kplex(F_set):
        return None

    best = [set(F_set)]

    # 初始候选集
    candidates = set()
    for v in G.nodes():
        if v in F_set:
            continue
        cnt = len(adj[v] & F_set)
        if cnt >= n_F + 1 - k:
            candidates.add(v)

    # 用栈模拟递归：每项是 (P, C)
    stack = [(set(F_set), candidates)]

    while stack:
        if time_limit is not None and _time.time() - start_time > time_limit:
            break

        P, C = stack.pop()

        # 更新最优
        if len(P) > len(best[0]):
            best[0] = set(P)

        if not C:
            continue

        # 剪枝
        upper = len(P) + len(C)
        if upper <= len(best[0]) or upper < bound:
            continue

        # 分支启发式：优先选与 P 连接最多的节点
        v = max(C, key=lambda x: len(adj[x] & P))
        C_rest = C - {v}

        # ---------- 分支 1：把 v 加入 P ----------
        can_add = True
        new_size = len(P) + 1

        # v 自身的度约束
        if len(adj[v] & P) < new_size - k:
            can_add = False

        # P 中已有节点的度约束
        if can_add:
            for u in P:
                if u in adj[v]:
                    cnt_u = len(adj[u] & P) + 1
                else:
                    cnt_u = len(adj[u] & P)
                if cnt_u < new_size - k:
                    can_add = False
                    break

        if can_add:
            P_new = P | {v}
            # 进一步过滤候选集：只有那些与 P_new 连接足够的节点才保留
            C_new = set()
            for x in C_rest:
                if len(adj[x] & P_new) >= len(P_new) + 1 - k:
                    C_new.add(x)
            stack.append((P_new, C_new))

        # ---------- 分支 2：不加入 v ----------
        stack.append((P, C_rest))

    if len(best[0]) >= bound:
        return list(best[0])
    return None


# =========================================================
# 主函数：BK 找最大团 -> BS 找最大 k-plex
# =========================================================
def runMain(k, lamda, loadG):
    """
    KPGN 社区种子创建阶段。

    流程：
        1. 用 k-core 剪枝去掉度不够的节点
        2. 循环：
            a. Bron-Kerbosch 找最大团
            b. 以最大团为约束集 F，用 BS 算法找最大 k-plex
            c. 记录并删除已用节点
        3. 写出种子社区到文件
    """
    G = copy.deepcopy(loadG)          # 工作图，会被删节点
    common = []

    # ---------- Step 1: k-core 剪枝 ----------
    def kCore(threshold, Gn):
        changed = True
        while changed:
            changed = False
            to_remove = [n for n in list(Gn.nodes()) if Gn.degree(n) < threshold]
            if to_remove:
                Gn.remove_nodes_from(to_remove)
                changed = True
        return Gn

    GACore = kCore(lamda - k, G)

    # ---------- Step 2: 主循环 ----------
    iteration = 0
    while GACore.number_of_nodes() >= lamda:
        iteration += 1
        cliques = list(nx.find_cliques(GACore))
        if not cliques:
            break

        max_clique = max(cliques, key=len)
        max_clique_size = len(max_clique)

        kplex = bs_max_kplex(GACore, k, max_clique, bound=lamda, time_limit=30)

        if kplex is None:
            if max_clique_size >= lamda:
                kplex = max_clique
            else:
                GACore.remove_nodes_from(max_clique)
                continue

        common.append(list(kplex))
        print(f"[Iter {iteration}] k-plex size = {len(kplex)}")
        GACore.remove_nodes_from(kplex)

    print("Number of communities:", len(common))

    return common


def seeds_to_labels(common, num_nodes):
    """
    common: [[node, node, ...], ...]
    返回:
        labels: [N] LongTensor，种子节点为 0..c-1，非种子节点为 -1
        c: 种子社区数
        seed_mask: [N] BoolTensor，True 表示该节点是种子节点
    """
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    for idx, seed in enumerate(common):
        for node in seed:
            labels[node] = idx
    c = len(common)
    seed_mask = labels >= 0
    return labels, c, seed_mask