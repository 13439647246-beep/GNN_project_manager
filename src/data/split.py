# ============================================================
# src/data/split.py
# 数据划分：为图数据生成训练/验证/测试掩码（基于 PyG）。
#
# 两种主流划分方式：
#   ① 公开固定划分：Planetoid 论文标准切分，保证结果可比
#   ② 随机划分    ：按比例或按类别采样，用于多次实验取平均
#
# 划分结果统一挂在 data 上的三个 bool 掩码：
#   data.train_mask / data.val_mask / data.test_mask
# ============================================================

import torch
from torch_geometric.transforms import RandomNodeSplit


# ============================================================
# ① 公开固定划分（Planetoid 标准）
# ============================================================

def public_split(name):
    """
    返回 Planetoid 官方标准划分的元信息。

    Planetoid 论文规定的切分规模（与复现结果对齐的关键）：
      Cora     : 140 训练 + 500 验证 + 1000 测试
      CiteSeer : 120 训练 + 500 验证 + 1000 测试
      PubMed   :  60 训练 + 500 验证 + 1000 测试

    注意：实际的掩码由 PyG 的 Planetoid 数据集在加载时自动生成，
    本函数只返回规模信息，便于日志或断言。

    参数:
        name : "cora" / "citeseer" / "pubmed"
    返回:
        dict: {"train": int, "val": int, "test": int}
    """
    table = {
        "cora":     {"train": 140, "val": 500, "test": 1000},
        "citeseer": {"train": 120, "val": 500, "test": 1000},
        "pubmed":   {"train": 60,  "val": 500, "test": 1000},
    }
    if name.lower() not in table:
        raise ValueError(f"未知数据集: {name}，可选 {list(table)}")
    return table[name.lower()]


# ============================================================
# ② 随机划分（对已加载的 data 事后处理）
# ============================================================

def random_split(data, num_train_per_class=20, num_val=500, num_test=1000):
    """
    随机划分：每类取 num_train_per_class 个节点作为训练，
    再随机取 num_val / num_test 个作为验证 / 测试。

    这在复现论文时很有用——很多工作报告的是"多次随机划分取平均"的结果。
    想跑多次实验时，每次调用都会重新随机采样。

    参数:
        data                : PyG Data 对象，需含 data.y
        num_train_per_class : 每类训练节点数
        num_val             : 验证节点数（可为 float 表示比例，如 0.2）
        num_test            : 测试节点数（同上）
    返回:
        修改后的 data，已附加 train_mask / val_mask / test_mask
    """
    transform = RandomNodeSplit(
        split="random",
        num_train_per_class=num_train_per_class,
        num_val=num_val,
        num_test=num_test,
    )
    return transform(data)


def train_rest_split(data, num_val=500, num_test=1000):
    """
    train_rest 模式：除验证/测试外全部用于训练。

    适合训练节点越多越好的场景（如半监督变体），或验证集划分固定时。
    """
    transform = RandomNodeSplit(split="train_rest", num_val=num_val, num_test=num_test)
    return transform(data)


def test_rest_split(data, num_train_per_class=20, num_val=500):
    """
    test_rest 模式：除训练/验证外全部用于测试。

    适合测试节点希望尽可能多的场景。
    """
    transform = RandomNodeSplit(
        split="test_rest",
        num_train_per_class=num_train_per_class,
        num_val=num_val,
    )
    return transform(data)


# ============================================================
# ③ 工具：掩码统计（用于日志、断言、调试）
# ============================================================

def mask_summary(data):
    """
    返回三种掩码的节点数，便于打印或断言。

    用法：
        print(mask_summary(data))
        # {"train": 140, "val": 500, "test": 1000}
    """
    return {
        "train": int(data.train_mask.sum()),
        "val":   int(data.val_mask.sum()),
        "test":  int(data.test_mask.sum()),
    }


def assert_public_split(data, name):
    """
    断言 data 上的掩码与 Planetoid 公开划分一致。
    加载数据后做一次校验，可以提早发现加载环节的静默错误。
    """
    expected = public_split(name)
    actual   = mask_summary(data)
    if actual != expected:
        raise AssertionError(
            f"{name} 掩码规模不符：期望 {expected}，实际 {actual}"
        )