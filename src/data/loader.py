# ============================================================
# src/data/loader.py
# 通用数据加载入口（基于 PyTorch Geometric）。
#
# 设计约定：
#   1. 所有数据集通过 register_data 注册，外部只认小写名字；
#   2. 每个数据集实际缓存在 <root>/<注册名>/ 下，和 _cache_dir_for 一致；
#   3. 加载失败时自动清理 processed/ 或整个缓存后重试；
#   4. 返回的 PyG Data 上挂 num_features / num_classes，供模型侧使用。
# ============================================================

import os
import shutil

import torch
from torch_geometric.datasets import (
    Planetoid,        # 引文网络：Cora / CiteSeer / PubMed
    Amazon,           # 商品共购：Computers / Photo
    Coauthor,         # 合作者网络：CS / Physics
    CitationFull,     # 完整引文网络：Cora / Cora_ML / CiteSeer / DBLP / PubMed
    KarateClub,       # 小型社交网络
    PolBlogs,         # 政治博客网络（无节点特征，需构造）
    WikipediaNetwork, # 维基页面网络：Chameleon / Squirrel / Crocodile
    Actor,            # 演员共现网络
    WikiCS,           # 维基计算机条目网络（多 split mask）
    WebKB,            # 网页分类：Cornell / Texas / Wisconsin
)

from src.utils.registry import register_data, build_data


# ============================================================
# 支持的数据集清单（仅作文档 / 报错提示用，不参与加载分派）
# 格式：(注册名, PyG 内部 name, 规模说明)
# ============================================================
PYG_NAMES = [
    # ---- Planetoid 引文网络（半监督节点分类基准）----
    ("cora",     "Cora",     "引文网络，2708 节点 / 10556 边 / 1433 维特征 / 7 类"),
    ("citeseer", "CiteSeer", "引文网络，3327 节点 / 9104 边 / 3703 维特征 / 6 类"),
    ("pubmed",   "PubMed",   "医学引文网络，19717 节点 / 88648 边 / 500 维特征 / 3 类"),

    # ---- Amazon 商品共购网络 ----
    ("amazon_computers", "Computers", "商品共购，13752 节点 / 491722 边 / 767 维特征 / 10 类"),
    ("amazon_photo",     "Photo",     "商品共购，7650 节点 / 238162 边 / 745 维特征 / 8 类"),

    # ---- Coauthor 合作者网络 ----
    ("coauthor_cs",      "CS",      "合作者网络，18333 节点 / 163788 边 / 6805 维特征 / 15 类"),
    ("coauthor_physics", "Physics", "合作者网络，34493 节点 / 495924 边 / 8415 维特征 / 5 类"),

    # ---- CitationFull 完整引文网络 ----
    ("citationfull_cora",     "Cora",    "完整 Cora，19793 节点 / 126842 边 / 8710 维特征 / 70 类"),
    ("citationfull_cora_ml",  "Cora_ML", "Cora_ML，2995 节点 / 16316 边 / 2879 维特征 / 7 类"),
    ("citationfull_citeseer", "CiteSeer","完整 CiteSeer，4230 节点 / 10674 边 / 602 维特征 / 6 类"),
    ("citationfull_dblp",     "DBLP",    "DBLP，17716 节点 / 105734 边 / 1639 维特征 / 4 类"),
    ("citationfull_pubmed",   "PubMed",  "完整 PubMed，19717 节点 / 88648 边 / 500 维特征 / 3 类"),

    # ---- KarateClub 小型社交网络 ----
    ("karate", None, "Zachary 空手道俱乐部，34 节点 / 156 边 / 34 维特征 / 4 类"),

    # ---- PolBlogs（密集、无节点特征）----
    ("polblogs", "PolBlogs", "政治博客网络，1490 节点 / 19025 边 / 度特征 / 2 类"),

    # ---- WikipediaNetwork（密集、有节点特征）----
    ("chameleon", "chameleon", "维基页面网络，2277 节点 / 62742 边 / 2325 维特征 / 5 类"),
    ("squirrel",  "squirrel",  "维基页面网络，5201 节点 / 396706 边 / 2089 维特征 / 5 类"),
    ("crocodile", "crocodile", "维基页面网络，11631 节点 / 360658 边 / 1316 维特征 / 5 类"),

    # ---- Actor 演员共现网络 ----
    ("actor", "Actor", "演员共现，7600 节点 / 53318 边 / 932 维特征 / 5 类"),

    # ---- WikiCS 维基计算机条目网络 ----
    ("wikics", "WikiCS", "维基计算机条目，11701 节点 / 431206 边 / 300 维特征 / 10 类"),

    # ---- WebKB 网页分类（规模较小，Qp 易饱和）----
    ("cornell",   "Cornell",   "网页分类，183 节点 / 295 边 / 1703 维特征 / 5 类"),
    ("texas",     "Texas",     "网页分类，183 节点 / 309 边 / 1703 维特征 / 5 类"),
    ("wisconsin", "Wisconsin", "网页分类，251 节点 / 499 边 / 1703 维特征 / 5 类"),
]


# ============================================================
# 内部工具：把 Data 对象的元信息补齐
# ============================================================
def _finalize_data(data, num_features=None, num_classes=None):
    """
    给 Data 挂上 num_features / num_classes，供模型和 describe_data 使用。

    参数:
        data         : PyG Data 对象
        num_features : 显式指定特征维度；None 时从 data.x 推断
        num_classes  : 显式指定类别数；None 时从 data.y 推断
    """
    if num_features is None:
        if getattr(data, "x", None) is not None:
            num_features = int(data.x.size(1))
        else:
            num_features = 0
    if num_classes is None:
        num_classes = int(data.y.max().item()) + 1

    data.num_features = int(num_features)
    data.num_classes = int(num_classes)
    return data


# ============================================================
# 构建函数：按 arg["data"]["name"] 分派到对应的 PyG Dataset
# ============================================================
def _build_dataset(arg):
    """
    根据 arg["data"]["name"] 构建对应的 PyG 数据集。

    统一约定：
        - 所有数据集缓存在 <root>/<注册名>/ 下；
        - 返回前补齐 num_features / num_classes；
        - 需要特殊处理的（PolBlogs 无特征、WikiCS 多列 mask）在分支内直接 return。
    """
    raw_name = arg["data"]["name"].lower()
    root = arg["data"].get("root", "data")

    # 每个数据集的独立缓存目录，保证 _cache_dir_for 能对上
    dataset_root = os.path.join(root, raw_name)

    # --------------------------------------------------------
    # Planetoid：Cora / CiteSeer / PubMed
    # --------------------------------------------------------
    if raw_name in ("cora", "citeseer", "pubmed"):
        pyg_name = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[raw_name]
        dataset = Planetoid(root=dataset_root, name=pyg_name)
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # Amazon：Computers / Photo
    # --------------------------------------------------------
    if raw_name in ("amazon_computers", "amazon_photo"):
        pyg_name = "Computers" if "computers" in raw_name else "Photo"
        dataset = Amazon(root=dataset_root, name=pyg_name)
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # Coauthor：CS / Physics
    # --------------------------------------------------------
    if raw_name in ("coauthor_cs", "coauthor_physics"):
        pyg_name = "CS" if raw_name.endswith("_cs") else "Physics"
        dataset = Coauthor(root=dataset_root, name=pyg_name)
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # CitationFull：Cora / Cora_ML / CiteSeer / DBLP / PubMed
    # --------------------------------------------------------
    if raw_name.startswith("citationfull_"):
        suffix = raw_name[len("citationfull_"):]
        # 把 citationfull_cora_ml → Cora_ML，citationfull_citeseer → CiteSeer
        pyg_name = suffix.replace("_", " ").title().replace(" ", "_")
        if pyg_name == "Cora_Ml":
            pyg_name = "Cora_ML"
        if pyg_name == "Citeseer":
            pyg_name = "CiteSeer"
        dataset = CitationFull(root=dataset_root, name=pyg_name)
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # KarateClub：无 train/val/test mask，仅用于快速调试
    # --------------------------------------------------------
    if raw_name in ("karate", "karateclub"):
        dataset = KarateClub()
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # PolBlogs：密集小图，无节点特征、无划分
    #   - 用节点度构造 1 维特征；
    #   - 按类别 20/500/剩余 随机划分。
    # --------------------------------------------------------
    if raw_name == "polblogs":
        dataset = PolBlogs(root=dataset_root)
        data = dataset[0]
        n = int(data.num_nodes)
        device = data.edge_index.device

        # 构造度特征（PolBlogs 原始没有 x）
        if getattr(data, "x", None) is None:
            deg = torch.zeros(n, 1, dtype=torch.float32, device=device)
            deg = deg.scatter_add(
                0,
                data.edge_index[0].unsqueeze(-1),
                torch.ones(data.edge_index.size(1), 1, dtype=torch.float32, device=device),
            )
            data.x = deg

        # 构造随机划分
        if getattr(data, "train_mask", None) is None:
            num_classes = int(data.y.max().item()) + 1
            perm = torch.randperm(n, device=device)
            n_train = min(20 * num_classes, n)
            n_val = min(500, n - n_train)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train + n_val]
            test_idx = perm[n_train + n_val:]

            data.train_mask = torch.zeros(n, dtype=torch.bool, device=device)
            data.val_mask = torch.zeros(n, dtype=torch.bool, device=device)
            data.test_mask = torch.zeros(n, dtype=torch.bool, device=device)
            data.train_mask[train_idx] = True
            data.val_mask[val_idx] = True
            data.test_mask[test_idx] = True

        return _finalize_data(data)

    # --------------------------------------------------------
    # WikipediaNetwork：Chameleon / Squirrel / Crocodile
    # --------------------------------------------------------
    if raw_name in ("chameleon", "squirrel", "crocodile"):
        dataset = WikipediaNetwork(root=dataset_root, name=raw_name)
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # Actor：演员共现网络
    # --------------------------------------------------------
    if raw_name == "actor":
        dataset = Actor(root=dataset_root)
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # WikiCS：train_mask 是 (N, 20)，取第一列作为单 split
    # --------------------------------------------------------
    if raw_name == "wikics":
        dataset = WikiCS(root=dataset_root)
        data = dataset[0]
        if getattr(data, "train_mask", None) is not None and data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]
        if getattr(data, "val_mask", None) is not None and data.val_mask.dim() > 1:
            data.val_mask = data.val_mask[:, 0]
        if getattr(data, "test_mask", None) is not None and data.test_mask.dim() > 1:
            data.test_mask = data.test_mask[:, 0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # WebKB：Cornell / Texas / Wisconsin
    # --------------------------------------------------------
    if raw_name in ("cornell", "texas", "wisconsin"):
        pyg_name = raw_name.capitalize()
        dataset = WebKB(root=dataset_root, name=pyg_name)
        data = dataset[0]
        return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

    # --------------------------------------------------------
    # 未知数据集：一次性列出所有可用名字
    # --------------------------------------------------------
    raise ValueError(
        f"不支持的 dataset: '{raw_name}'。\n"
        f"可用的名字（不区分大小写）：\n"
        f"  Planetoid:    cora, citeseer, pubmed\n"
        f"  Amazon:       amazon_computers, amazon_photo\n"
        f"  Coauthor:     coauthor_cs, coauthor_physics\n"
        f"  CitationFull: citationfull_cora, citationfull_cora_ml, "
        f"citationfull_citeseer, citationfull_dblp, citationfull_pubmed\n"
        f"  KarateClub:   karate\n"
        f"  PolBlogs:     polblogs\n"
        f"  Wikipedia:    chameleon, squirrel, crocodile\n"
        f"  Actor:        actor\n"
        f"  WikiCS:       wikics\n"
        f"  WebKB:        cornell, texas, wisconsin"
    )


# ============================================================
# 缓存清理：与 dataset_root 保持一致
# ============================================================
def _cache_dir_for(arg):
    """
    返回当前数据集对应的缓存根目录。

    因为 _build_dataset 里所有数据集都用 <root>/<注册名>/，
    这里直接返回同一路径，清理时不会误删其他数据集。
    """
    root = arg["data"].get("root", "data")
    return os.path.join(root, arg["data"]["name"].lower())


def _clean_processed(arg):
    """只清 processed/，保留已下载的 raw/（raw 完好时能省一次下载）"""
    proc = os.path.join(_cache_dir_for(arg), "processed")
    if os.path.exists(proc):
        shutil.rmtree(proc, ignore_errors=True)
        print(f"[data] 清理损坏的 processed 缓存：{proc}")


def _clean_all(arg):
    """清整个数据集目录（raw + processed），用于 raw 也坏了的情况"""
    cache = _cache_dir_for(arg)
    if os.path.exists(cache):
        shutil.rmtree(cache, ignore_errors=True)
        print(f"[data] 清理整个数据目录：{cache}")


def build_dataset_with_retry(arg):
    """
    加载数据集，带两级自动恢复：
      ① 失败 → 清 processed/ → 重试
      ② 再失败 → 清整个目录（含 raw）→ 重下 → 再试一次
      ③ 还失败 → 抛原始异常，让调用方看到
    """
    try:
        return _build_dataset(arg)
    except Exception as e1:
        print(f"[data] 首次加载失败：{type(e1).__name__}: {e1}")
        _clean_processed(arg)

    try:
        return _build_dataset(arg)
    except Exception as e2:
        print(f"[data] 重试仍失败，清空缓存重新下载：{type(e2).__name__}: {e2}")
        _clean_all(arg)

    return _build_dataset(arg)  # 最后一次尝试，失败就让它抛


# ============================================================
# 注册所有支持的数据集
# ============================================================
# Planetoid
register_data("cora")(build_dataset_with_retry)
register_data("citeseer")(build_dataset_with_retry)
register_data("pubmed")(build_dataset_with_retry)

# Amazon（按需开启，下载量较大）
# register_data("amazon_computers")(build_dataset_with_retry)
# register_data("amazon_photo")(build_dataset_with_retry)

# Coauthor
register_data("coauthor_cs")(build_dataset_with_retry)
register_data("coauthor_physics")(build_dataset_with_retry)

# CitationFull
register_data("citationfull_cora")(build_dataset_with_retry)
register_data("citationfull_cora_ml")(build_dataset_with_retry)
register_data("citationfull_citeseer")(build_dataset_with_retry)
register_data("citationfull_dblp")(build_dataset_with_retry)
register_data("citationfull_pubmed")(build_dataset_with_retry)

# KarateClub
register_data("karate")(build_dataset_with_retry)

# PolBlogs（密集、无特征）
register_data("polblogs")(build_dataset_with_retry)

# WikipediaNetwork（密集、有特征）
register_data("chameleon")(build_dataset_with_retry)
register_data("squirrel")(build_dataset_with_retry)
register_data("crocodile")(build_dataset_with_retry)

# Actor
register_data("actor")(build_dataset_with_retry)

# WikiCS
register_data("wikics")(build_dataset_with_retry)

# WebKB
register_data("cornell")(build_dataset_with_retry)
register_data("texas")(build_dataset_with_retry)
register_data("wisconsin")(build_dataset_with_retry)


# ============================================================
# 对外主入口
# ============================================================
def load_data(name, arg=None):
    """
    统一数据加载接口。

    参数:
        name : 数据集名（不区分大小写），如 "cora"、"polblogs"、"chameleon"
        arg  : 可选，全局 arg dict。读 arg['data']['root'] 作为数据根目录。

    返回:
        PyG Data 对象（含 x / y / edge_index / train_mask / val_mask / test_mask）
    """
    arg = dict(arg or {})
    data_cfg = dict(arg.get("data", {}))
    data_cfg["name"] = name
    arg["data"] = data_cfg
    return build_data(arg)


def describe_data(data, name=None):
    """
    打印 / 返回数据集的形状与规模信息，用于加载后快速核对。

    参数:
        data : PyG Data 对象（需含 x / y / edge_index）
        name : 可选，数据集名，仅用于打印标题

    返回:
        dict: 各项信息，方便程序里继续使用（例如断言或写日志）
    """
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))
    num_features = int(data.x.size(1)) if getattr(data, "x", None) is not None else 0

    # 类别数优先用 loader 挂上去的 num_classes；没有就从 y 推断
    num_classes = int(getattr(data, "num_classes", data.y.max().item() + 1))

    def _mask_size(mask_name):
        m = getattr(data, mask_name, None)
        return int(m.sum()) if m is not None else None

    info = {
        "name": name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": num_features,
        "num_classes": num_classes,
        "train": _mask_size("train_mask"),
        "val": _mask_size("val_mask"),
        "test": _mask_size("test_mask"),
    }

    title = f"[data] {name}" if name else "[data]"
    print(f"{title}")
    print(f"  nodes={num_nodes}  edges={num_edges}  "
          f"features={num_features}  classes={num_classes}")
    print(f"  split: train={info['train']}  val={info['val']}  test={info['test']}")

    return info