# src/data/ — 通用数据加载与管理

本模块负责**所有数据集的下载、解析、预处理与划分**。基于 PyTorch Geometric (PyG) 构建，自动处理数据缓存、异常重试，并通过统一的接口向上层（模型、训练引擎）提供标准化的数据格式。

---

## 一、目录结构

```text
src/data/
├── __init__.py       # 对外接口：load_data, prepare_data, describe_data 及各类 transform/split 导出
├── loader.py         # 数据加载核心：自动下载、缓存、异常恢复、特殊数据集处理
├── transform.py      # 通用预处理：特征归一化、图结构归一化、层内 Norm 工厂
└── split.py          # 数据划分：公开划分、随机划分、掩码统计与校验
```

---

## 二、核心接口与使用规范

### 1. 数据加载与元信息注入（模型首选）
所有的模型和训练流程都应该使用 `prepare_data` 作为唯一入口。它不仅加载数据，还会把模型的必要元信息写入 `arg` 字典中。

```python
from src.data import prepare_data, describe_data

# 加载、归一化并注入元信息
data = prepare_data("cora", arg)

# 此时 arg 中已自动写入：
# arg["data"]["num_features"] = 1433
# arg["data"]["num_classes"]  = 7
# arg["data"]["num_nodes"]    = 2708

# 打印数据摘要
describe_data(data, "cora")
```
*注意：`prepare_data` 内部默认调用了 `normalize_features_l1`（L1 行归一化）。*

### 2. 底层加载与容错
如果需要自行控制预处理，可以直接调用 `load_data`。它内部通过 `build_dataset_with_retry` 包装，**自带三级容错机制**：
1. 加载失败 → 清空 `processed/` 缓存并重试
2. 再次失败 → 清空整个数据集目录（含 `raw/`）并重新下载
3. 仍然失败 → 抛出原始异常

```python
from src.data import load_data
data = load_data("chameleon", arg)
```

---

## 三、支持的数据集列表

调用时请使用**注册名**（全小写）。所有数据均统一缓存在 `data/<注册名>/` 目录下。

### 1. 引文网络 (Planetoid / CitationFull)
| 注册名                         | 规模 (节点/边/特征/类别) | 说明                   |
|--------------------------------|--------------------------|------------------------|
| `cora` / `citeseer` / `pubmed` | 2.7k~19.7k 节点          | 经典半监督节点分类基准 |
| `citationfull_cora`            | 19.8k / 126k / 8710 / 70 | 完整的 Cora 引文网络   |
| `citationfull_cora_ml`         | 3.0k / 16k / 2879 / 7    | Cora_ML 版本           |
| `citationfull_citeseer`        | 4.2k / 10k / 602 / 6     | 完整 CiteSeer          |
| `citationfull_dblp`            | 17.7k / 105k / 1639 / 4  | DBLP 文献网络          |
| `citationfull_pubmed`          | 19.7k / 88k / 500 / 3    | 完整 PubMed            |

### 2. 合作者与商品网络 (Coauthor / Amazon)
| 注册名             | 规模 (节点/边/特征/类别) | 说明                             |
|--------------------|--------------------------|----------------------------------|
| `coauthor_cs`      | 18.3k / 163k / 6805 / 15 | 计算机科学合作者                 |
| `coauthor_physics` | 34.5k / 495k / 8415 / 5  | 物理学合作者                     |
| `amazon_computers` | 13.7k / 491k / 767 / 10  | 商品共购（*需在代码中取消注册*） |
| `amazon_photo`     | 7.6k / 238k / 745 / 8    | 商品共购（*需在代码中取消注册*） |

### 3. 异质图与网页网络 (Wikipedia / WebKB / Actor)
| 注册名                                 | 规模 (节点/边/特征/类别) | 说明                     |
|----------------------------------------|--------------------------|--------------------------|
| `chameleon` / `squirrel` / `crocodile` | 2.2k~11.6k 节点          | 维基页面网络，边极密集   |
| `actor`                                | 7.6k / 53k / 932 / 5     | 演员共现网络             |
| `cornell` / `texas` / `wisconsin`      | 183~251 节点             | WebKB 网页分类，规模极小 |

### 4. 其他特殊网络 (WikiCS / PolBlogs / Karate)
| 注册名     | 规模 (节点/边/特征/类别) | 说明                                   |
|------------|--------------------------|----------------------------------------|
| `wikics`   | 11.7k / 431k / 300 / 10  | 维基计算机条目（原版多列 split mask）  |
| `polblogs` | 1490 / 19025 / 1 / 2     | 政治博客网络（**原版无节点特征**）     |
| `karate`   | 34 / 156 / 34 / 4        | Zachary 空手道俱乐部（**原版无划分**） |

---

## 四、特殊数据集的自动处理策略

`loader.py` 内部对几个不符合标准的 PyG 数据集做了自动兼容，保证上层模型不需要写 `if-else`：

1. **WikiCS**：原版返回的 `train_mask` 是一个 `(N, 20)` 的多维张量。代码会自动截取**第一列**，变成标准的 `(N,)` 布尔掩码。
2. **PolBlogs**：原版没有节点特征（`data.x` 为 `None`），代码会自动**提取节点度数**构造 `(N, 1)` 维特征；并且自动生成 20/500/剩余的随机 train/val/test 划分。
3. **无划分数据集**：`Karate`、`Actor`、`WebKB`、`WikipediaNetwork` 等在 PyG 中默认无 mask。若训练时报错缺失 mask，需在模型或引擎中手动调用 `split.py` 下的 `random_split` 进行划分。

---

## 五、扩展新的数据集

若要添加本模块尚未支持的 PyG 数据集，请在 `loader.py` 中执行两步：

1. **在顶部导入对应的 PyG 类**：
   ```python
   from torch_geometric.datasets import Flickr
   ```
2. **在 `_build_dataset` 中添加分支并注册**：
   ```python
   if raw_name == "flickr":
       dataset = Flickr(root=dataset_root)
       data = dataset[0]
       return _finalize_data(data, dataset.num_node_features, dataset.num_classes)

   # 在文件底部注册
   register_data("flickr")(build_dataset_with_retry)
   ```

---

## 六、常见问题与注意事项

1. **网络中断与重试**：国内直连 GitHub 极易报 `EOFError`。本模块的 `build_dataset_with_retry` 会自动清理残缺文件重试；若反复失败，请手动下载原始文件放入 `data/<注册名>/raw/` 目录。
2. **逻辑错误不重试**：`ValueError`（如数据集名字拼错）会直接抛出，不触发重试机制，避免无意义的清空操作。
3. **路径与缓存映射**：所有数据实际缓存在 `<root>/<注册名>/` 下（如 `data/cora/`，而不是 `data/Planetoid/Cora/`），便于直接按数据集名称清理缓存。
4. **内存开销**：`coauthor_physics`、`citationfull_cora`、`squirrel` 等图节点数上万且边极密集。在构造稠密矩阵（如 (N, N) 的拉普拉斯矩阵或模块度矩阵）时，请注意显存占用（数万节点的稠密矩阵极易 OOM），尽量使用 `torch.sparse.mm`。