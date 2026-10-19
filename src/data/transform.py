# ============================================================
# src/data/transform.py
# 图数据预处理：归一化相关操作集合（基于 PyG）。
#
# 归一化分三个层面，本文件按层面分类：
#   ① 节点特征归一化    —— 作用于 data.x
#   ② 图结构归一化      —— 作用于邻接矩阵 / 拉普拉斯矩阵
#   ③ 网络层归一化      —— 插入 GNN 层之间的归一化模块（nn.Module）
# ============================================================

import torch
from torch_geometric.transforms import NormalizeFeatures, NormalizeScale
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GraphNorm, PairNorm, LayerNorm


# ============================================================
# ① 节点特征归一化（作用于 data.x，返回修改后的 data）
# ============================================================

def normalize_features_l1(data):
    """
    L1 行归一化：每一行除以该行的和，使每行和为 1。

    这是 Planetoid 论文对词袋特征的标配预处理，
    Cora / CiteSeer / PubMed 复现时必用。

    参数:
        data : PyG Data 对象，需含 data.x
    返回:
        修改后的 data（原地修改，x 已被替换）
    """
    return NormalizeFeatures(attrs=["x"])(data)


def normalize_features_l2(data):
    """
    L2 行归一化：每一行除以其 L2 范数，使每行是单位向量。

    适用于需要数值稳定或距离度量的场景，
    一般不用于 Planetoid 系列（会破坏词袋频率语义）。

    参数:
        data : PyG Data 对象，需含 data.x
    返回:
        修改后的 data
    """
    return NormalizeScale(attrs=["x"])(data)


# ============================================================
# ② 图结构归一化（作用于邻接矩阵 / 拉普拉斯矩阵）
# ============================================================

def normalized_laplacian(edge_index, edge_weight=None, num_nodes=None,
                         normalization="sym"):
    """
    计算归一化拉普拉斯矩阵的稀疏表示。

    三种 normalization 取值：
      - None : L = D - A                （原始组合拉普拉斯）
      - "sym": L = I - D^{-1/2} A D^{-1/2}  （对称归一化，GCN 原论文同款）
      - "rw" : L = I - D^{-1} A          （随机游走归一化，按行归一化）

    参数:
        edge_index    : (2, E) 边索引
        edge_weight   : (E,) 边权重，None 时视为全 1
        num_nodes     : 节点数，None 时从 edge_index 推断
        normalization : "sym" / "rw" / None
    返回:
        (edge_index_lap, edge_weight_lap)：拉普拉斯矩阵的稀疏表示
    """
    return get_laplacian(
        edge_index,
        edge_weight=edge_weight,
        normalization=normalization,
        num_nodes=num_nodes,
    )


def gcn_normalized_adj(edge_index, edge_weight=None, num_nodes=None,
                       add_self_loops=True):
    """
    GCN 论文中的归一化邻接矩阵 D^{-1/2} A D^{-1/2}。

    这是 GCNConv 内部实际使用的形式，与"拉普拉斯归一化"等价但符号相反
    （GCN 用 A 而非 L，所以是加号形式）。

    参数:
        edge_index     : (2, E) 边索引
        edge_weight    : (E,) 边权重
        num_nodes      : 节点数
        add_self_loops : 是否加自环（GCN 原文默认加）
    返回:
        (edge_index_norm, edge_weight_norm)：归一化邻接矩阵的稀疏表示

    典型用法：
        edge_index_n, edge_weight_n = gcn_normalized_adj(data.edge_index)
        # 后续可以自己写卷积：
        #   out = edge_weight_n.view(-1,1) * x[edge_index_n[1]]
        #   out = scatter_add(out, edge_index_n[0], dim=0, dim_size=N)
    """
    return gcn_norm(
        edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        add_self_loops=add_self_loops,
    )


# ============================================================
# ③ 网络层归一化（返回 nn.Module，插入到 GNN 层之间）
# ============================================================
# 注意：这一类不是"数据预处理"，而是模型内部的归一化模块，
# 这里放在一起只是为了让归一化相关工具集中管理。
# 实际使用时由模型（models/<name>/model.py）创建并插入到 forward 中。
# ============================================================

def make_graph_norm(num_features):
    """
    GraphNorm：在 InstanceNorm 基础上为均值增加可学习因子。

    缓解图数据小批量下 BN 失效的问题，适合图分类和小图任务。
    """
    return GraphNorm(num_features)


def make_pair_norm(scale=1.0):
    """
    PairNorm：通过对节点特征做中心化，缓解深层 GNN 的过平滑。

    常用于 GCN/GAT 堆叠很深时（>4 层）防止所有节点表示收敛到相同值。
    """
    return PairNorm(scale_individually=False, scale=scale)


def make_layer_norm(mode="node"):
    """
    LayerNorm：对特征做层归一化。
      - mode="node" : 对每个节点的特征向量归一化
      - mode="graph": 对整个图的所有节点特征一起归一化
    """
    return LayerNorm(mode=mode)