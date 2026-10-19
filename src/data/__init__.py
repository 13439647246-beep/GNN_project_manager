# ============================================================
# src/data/__init__.py
# 数据模块对外接口。
# 调用方只需 from src.data import load_data 即可。
# ============================================================

# ---- 加载 ----
from src.data.loader import load_data, describe_data

# ---- 归一化 / 预处理 ----
from src.data.transform import (
    normalize_features_l1,
    normalize_features_l2,
    normalized_laplacian,
    gcn_normalized_adj,
    make_graph_norm,
    make_pair_norm,
    make_layer_norm,
)

# ---- 划分 ----
from src.data.split import (
    public_split,
    random_split,
    train_rest_split,
    test_rest_split,
    mask_summary,
    assert_public_split,
)

__all__ = [
    # 加载
    "load_data", "describe_data",
    # 归一化
    "normalize_features_l1", "normalize_features_l2",
    "normalized_laplacian", "gcn_normalized_adj",
    "make_graph_norm", "make_pair_norm", "make_layer_norm",
    # 划分
    "public_split", "random_split", "train_rest_split", "test_rest_split",
    "mask_summary", "assert_public_split",
    # 准备
    "prepare_data",
]


def prepare_data(name, arg):
    """
    一步到位地准备数据：
      1) 加载（PyG Data）
      2) L1 归一化
      3) 把 num_features / num_classes 注入 arg["data"]，供模型构建使用
    返回 PyG Data 对象。
    """
    data = load_data(name, arg)
    data = normalize_features_l1(data)
    arg["data"]["num_features"] = data.num_features
    arg["data"]["num_classes"]  = data.num_classes
    arg["data"]["num_nodes"] = data.num_nodes
    return data