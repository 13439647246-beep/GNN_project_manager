# NN_project_manager

配置驱动的模型复现管理框架。用一份固定的 `arg` 字典贯穿模型构建、训练与网格搜索；用注册表把训练引擎和具体模型解耦；每个模型只写三个文件（`layer.py` / `model.py` / `loss.py`），其余通用能力全部上提到 `src/`。

---

## 1. 核心设计理念

| 理念                      | 含义                                                                    | 带来的好处                               |
|---------------------------|-------------------------------------------------------------------------|------------------------------------------|
| **固定 arg 字典驱动一切** | 模型构建、训练、损失、数据、网格搜索全部从同一个嵌套字典 `arg` 读取参数 | 配置即实验，换实验只换配置，不改代码     |
| **注册表解耦**            | `engine` 只通过名字构建对象，从不 `import` 具体模型                     | 新增模型不触碰训练代码，框架与模型零耦合 |
| **模型三文件制**          | 每个模型只写 `layer.py` / `model.py` / `loss.py`                        | 模型实现边界清晰，复制即新建             |
| **通用能力上提**          | 数据加载、图/注意力层、无状态运算、训练引擎统一放在 `src/`              | 模型之间不重复造轮子                     |
| **不用命令行**            | 根目录 `run_test.py` 改顶部变量切换实验                                 | 复现实验无需记忆长参数                   |

---

## 2. 整体架构

### 2.1 分层与依赖方向

```
┌──────────────────────────────────────────────────────┐
│  run_test.py            唯一实验入口（改顶部变量）   │
└──────────────────────────┬───────────────────────────┘
                           │ 1. 定位/生成 yaml  2. 合并出 arg
┌──────────────────────────▼───────────────────────────┐
│  src/config/            配置层                       │
│  DEFAULT_ARG → _base.yaml → <model>.yaml → overrides │
│  load_arg / validate_arg / ensure_model_yaml         │
└──────────────────────────┬───────────────────────────┘
                           │ arg（含 prepare_data 注入的元信息）
┌──────────────────────────▼───────────────────────────┐
│  src/utils/registry.py  注册表（解耦中枢）           │
│  MODEL_REGISTRY / LOSS_REGISTRY / DATA_REGISTRY      │
│  build_model / build_loss / build_data               │
└──────┬───────────────────┬───────────────────┬───────┘
       │                   │                   │
┌──────▼──────┐   ┌────────▼────────┐   ┌──────▼───────┐
│  models/    │   │  src/data/      │   │ src/engine/  │
│  三文件实现 │   │  加载/变换/划分 │   │ 训练编排     │
└──────┬──────┘   └─────────────────┘   └──────┬───────┘
       │ 复用                                  │ 复用
┌──────▼───────────────────────────────────────▼───────┐
│  src/layers/（通用网络层）  src/ops/（无状态运算库） │
└──────────────────────────────────────────────────────┘
```

**依赖规则（重要）：**

- `src/engine/` **不 import** `models/`，只通过 `build_model(arg)` 拿实例；
- `src/config/` 不依赖模型实现；
- `src/ops/`、`src/layers/` 是最底层，不依赖上层；
- 依赖方向始终向下，出现反向 import 即视为架构破坏。

### 2.2 单次运行调用链

```
run_test.main()
 ├─ ensure_model_yaml(MODEL_NAME, data_name=DATA_NAME)   # 无 yaml 则由 _template.yaml 生成
 ├─ load_arg(cfg_path, overrides={model.name, data.name})# 四层合并
 ├─ validate_arg(arg)                                    # 缺字段一次性报全
 ├─ data = prepare_data(arg["data"]["name"], arg)        # 加载 + L1 归一化 + 注入元信息
 ├─ describe_data(data, arg["data"]["name"])             # 打印节点/边/类别/划分统计
 └─ MODE == "single" → train_once(arg, data)
    MODE == "grid"   → run_grid(arg, GRID, data)
```

### 2.3 训练内部数据流

```
train_once(arg, data)
 ├─ build_model(arg) / build_loss(arg)        # 走注册表，不 import 具体模型
 ├─ prepare_input(data)                        # 若模型定义，训练前调用一次并缓存
 ├─ Trainer.fit()
 │   ├─ 每个 epoch 对 train/val 调 _evaluate_split(split)
 │   │    └─ hasattr(model, "evaluate") → 优先走模型自定义评估
 │   ├─ _compute_loss(outputs)                # 按 criterion.forward 参数个数分派
 │   │    ├─ nargs == 1 → criterion(outputs)
 │   │    └─ nargs >= 2 → criterion(outputs, data)
 │   ├─ Logger.log_epoch()                     # 控制台 + metrics.csv
 │   └─ Checkpoint.save_best/save_last()       # 依据 save.monitor / save.mode
 └─ trainer.test(best.pt) → 返回结果 dict
```

### 2.4 目录结构（含逐项注释）

```
project_root/
├── src/
│   ├── data/                        # 数据加载 + 预处理
│   │   ├── __init__.py              # 导出 load_data / describe_data / prepare_data 及各 transform/split
│   │   ├── loader.py                # PyG 数据集加载，内部走 registry 的 build_data
│   │   ├── transform.py             # 归一化：L1/L2、拉普拉斯、GCN norm、层内 norm 工厂
│   │   └── split.py                 # public / random / train_rest / test_rest 划分 + 掩码统计
│   ├── layers/                      # 通用网络层（可被所有模型复用）
│   │   ├── basic.py                 # get_activation / MLP / ResidualBlock
│   │   ├── graph.py                 # GNN
│   │   ├── attention.py             # MultiHeadSelfAttention（activation 可选，None 跳过）
│   │   └── initialization.py        # apply_init / xavier / kaiming / orthogonal / normal
│   ├── ops/                         # 无状态运算库（纯函数，不持有参数）
│   │   ├── functional.py            # softmax / masked_* / 各类 loss / scatter_softmax
│   │   ├── linalg.py                # 相似度 / 距离 / symmetric_normalize_adj / modularity
│   │   ├── metrics.py               # accuracy / top_k / P-R-F1 / MSE / MAE / R² / AUC
│   │   └── utils.py                 # to_tensor / one_hot / count_parameters / flatten_dict
│   ├── engine/                      # 训练编排（不 import 具体模型）
│   │   ├── logger.py                # Logger：控制台 + train.log + metrics.csv
│   │   ├── checkpoint.py            # Checkpoint：save_last / save_best / load
│   │   ├── evaluator.py             # evaluate(model, data, mask, criterion) 默认评估
│   │   ├── callbacks.py             # # 训练回调：EarlyStopping、最佳模型保存等
│   │   ├── run_layout.py            # 运行目录布局、网格命名、最优模型汇总
│   │   ├── trainer.py               # Trainer + _evaluate_split + _compute_loss
│   │   └── runner.py                # train_once + run_grid + print_summary + save_summary
│   ├── config/                      # 配置层
│   │   ├── default_arg.py           # DEFAULT_ARG 全局骨架
│   │   ├── arg_io.py                # load_arg / deep_merge / set_nested / ensure_model_yaml
│   │   └── grid.py                  # expand_grid（点号路径 → 笛卡尔积）
│   └── utils/                       # 基础设施
│       ├── registry.py              # 三表 + 三个装饰器 + build_* + list_*
│       ├── seed.py                  # set_seed
│       └── device.py                # get_device / move_to_device
├── models/
│   ├── __init__.py                  # 统一 import 所有模型（触发注册）
│   └── <name>/                      # 每个模型一个文件夹
│       ├── __init__.py              # 导入 model/loss 触发注册（★ 最容易漏）
│       ├── layer.py                 # 模型私有层
│       ├── model.py                 # @register_model 构建函数 + nn.Module
│       ├── loss.py                  # @register_loss 构建函数 + 损失
│       └── README.md                # 模型说明（数据、超参、指标）
├── configs/
│   ├── _base.yaml                   # 全局默认（所有模型继承）
│   ├── _template.yaml               # 新模型骨架（ensure_model_yaml 用它）
│   ├── <name>.yaml                  # 每个模型一份
│   └── grid/                        # 网格搜索配置
├── runs/                            # 实验产物（按模型/数据集/实验/时间戳隔离）
│   └── <model>/<timestamp>/<dataset>/<exp>/
│       ├── config.yaml              # 本次运行的完整 arg 快照
│       ├── train.log
│       ├── metrics.csv
│       └── ckpt/
│           ├── best.pt
│           └── last.pt
├── data/                            # PyG 数据集下载目标
├── tests/                           # 测试
├── requirements.txt
├── pyproject.toml
├── README.md
└── run_test.py                      # ★ 唯一实验入口（改顶部变量）
```

---

## 3. 运行方式

只改 `run_test.py` 顶部四个变量：

```python
MODEL_NAME = "gcn"       # 对应 models/<name>/ 和 configs/<name>.yaml
DATA_NAME  = "cora"      # 对应 src/data 注册名
MODE       = "single"    # "single" / "grid"
GRID       = {...}       # MODE="grid" 时使用，点号路径 → 候选值列表
```

然后 `python run_test.py`。

- `single`：跑一组配置，结果落在 `runs/<model>/<dataset>/<exp>/<timestamp>/`；
- `grid`：`expand_grid` 生成笛卡尔积，每个组合独立目录，汇总写入 `runs/<model>/<dataset>/_grid/summary_<ts>.csv`。

---

## 4. 配置系统

### 4.1 四层合并

```
DEFAULT_ARG  →  configs/_base.yaml  →  configs/<name>.yaml  →  overrides
   (骨架)          (全局默认)             (模型专属)          (点号路径覆盖)
```

`load_arg(cfg_path, overrides=None, base_path="configs/_base.yaml")` 完成合并；`overrides` 用点号路径，如 `{"optim.lr": 0.001}`。

### 4.2 YAML 约定

- `5e-4` 在 YAML 1.1 会被解析成**字符串**，请写 `0.0005` 或 `5.0e-4`；
- 模型专属超参放 `model.kwargs`，损失专属放 `loss.kwargs`；
- `save.monitor` 必须与模型实际产出的指标名一致，否则 `best.pt` 永不保存；
- `save.monitor` / `save.mode` 不是超参，不要放进 `GRID`。

### 4.3 保存策略

```yaml
# 节点分类
save:
  monitor: val_acc
  mode: max
```

```yaml
# 社区发现
save:
  monitor: val_Qp
  mode: max
```

---

## 5. 关键接口约定

### 5.1 注册表 `src/utils/registry.py`

| 接口                                                                   | 说明                          |
|------------------------------------------------------------------------|-------------------------------|
| `register_model(name)` / `register_loss(name)` / `register_data(name)` | 装饰器，注册构建函数          |
| `build_model(arg)`                                                     | `arg["model"]["name"]` → 实例 |
| `build_loss(arg)`                                                      | `arg["loss"]["name"]` → 实例  |
| `build_data(arg)`                                                      | `arg["data"]["name"]` → 实例  |
| `list_models()` / `list_losses()` / `list_datasets()`                  | 枚举已注册项                  |

### 5.2 数据 `src/data/`

| 接口                        | 说明                                                                                                                    |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `load_data(name, arg)`      | → PyG `Data`，内部走 `build_data`                                                                                       |
| `describe_data(data, name)` | 打印节点/边/特征/类别/划分                                                                                              |
| `prepare_data(name, arg)`   | 加载 + L1 归一化 + 向 `arg["data"]` 注入 `num_features` / `num_classes` / `num_nodes`；**单次训练与网格搜索都必须先调** |

已注册数据集（小写）：`cora` / `citeseer` / `pubmed` / `amazon_computers` / `amazon_photo` / `coauthor_cs` / `coauthor_physics` / `citationfull_{cora, cora_ml, citeseer, dblp, pubmed}` / `karate`。

### 5.3 引擎 `src/engine/`

| 对象                                                            | 主要方法                                                                                       |
|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `Logger(arg)`                                                   | `info` / `warning` / `log_epoch(epoch, metrics)` / `run_dir`                                   |
| `Checkpoint(arg, logger)`                                       | `save_last` / `save_best` / `load(path, model, optimizer, device)` / `best_path` / `last_path` |
| `Trainer(arg, model, data, criterion, optimizer, ckpt, logger)` | `fit()` / `test(ckpt_path)` / `_evaluate_split(split)` / `_compute_loss(outputs)`              |
| `runner`                                                        | `train_once(arg, data)` / `run_grid(base_arg, grid, data)` / `print_summary` / `save_summary`  |

`_evaluate_split` 优先使用模型自定义的 `evaluate`；`_compute_loss` 按损失 `forward` 的参数个数分派（`nargs==1` 走 `criterion(outputs)`，`nargs>=2` 走 `criterion(outputs, data)`）。

### 5.4 运算边界

| 场景                                | 放置位置                                                 |
|-------------------------------------|----------------------------------------------------------|
| 通用损失（CE / MSE / InfoNCE / KL） | `src/ops/functional.py`                                  |
| 模型特定损失（带正则、特定加权）    | `models/<name>/loss.py`，可调用 `ops` 中的函数           |
| 无状态图/线性代数运算               | `src/ops/linalg.py`                                      |
| 带参数的层                          | `src/layers/`（通用）或 `models/<name>/layer.py`（私有） |

---

## 6. 模型三文件规范

```
models/<name>/
├── __init__.py     # ★ 必须 import model 与 loss，否则注册不触发
├── layer.py        # 模型私有层
├── model.py        # @register_model + nn.Module（prepare_input / forward / evaluate）
├── loss.py         # @register_loss + nn.Module（forward 统一 (outputs, data=None)）
└── README.md
```

`__init__.py` 固定写法：

```python
from models.<name> import model   # 触发 @register_model
from models.<name> import loss    # 触发 @register_loss
```

`model.py` 关键约定：

- `__init__` 从 `arg["data"]` 读取 `num_features` / `num_classes` / `num_nodes`（由 `prepare_data` 注入）；
- 需要缓存张量时用 `register_buffer(..., persistent=False)`，判空用 `numel() == 0`；
- `prepare_input(data)` 训练前调用一次，必须显式写回 `self._cached_x = ...`；
- `forward` 里做兜底：缓存为空则自动调 `prepare_input`；
- `forward` 返回 dict（如 `{"logits": ...}`）或张量；返回 dict 时需提供 `evaluate`；
- 非分类任务必须自定义 `evaluate(data, split)`，返回包含 `save.monitor` 对应键的 dict。

`loss.py` 签名统一为：

```python
def forward(self, outputs, data=None):
    ...
    return loss, {"part_a": float(a), "part_b": float(b)}
```

---

## 7. 已实现模型

| 模型   | 类型     | 输出                                                        | monitor   | 评估方式            |
|--------|----------|-------------------------------------------------------------|-----------|---------------------|
| GCN    | 节点分类 | `{"logits": ...}` 或张量                                    | `val_acc` | 默认 evaluate       |
| GCNRH  | 社区发现 | `{"C", "mu", "beta", "B_o"}`                                | `val_Qp`  | 模型自定义 evaluate |
| KPGN   | 节点分类 | `{"edge_index", "idx_train", "output_train", "output_all"}` | `val_acc` | 模型自定义 evaluate |
| HGCNRH | 社区发现 | `{"C", "mu", "beta", "B_o"}`                                | `val_Qp`  | 模型自定义 evaluate |

---

## 8. 新增模型检查清单

1. 建目录 `models/<name>/`，写 `layer.py` / `model.py` / `loss.py` / `README.md`；
2. `models/<name>/__init__.py` 中 import `model` 与 `loss`（**最容易漏**）；
3. 在 `models/__init__.py` 增加 `from models import <name>`；
4. 执行 `ensure_model_yaml("<name>", data_name=...)` 生成 `configs/<name>.yaml`；
5. 在 yaml 中确认 `save.monitor` / `save.mode` 与模型实际指标名一致；
6. 若为非分类任务，在 `model.py` 中实现 `evaluate`；
7. 改 `run_test.py` 顶部 `MODEL_NAME` / `DATA_NAME`，先 `single` 跑通再上 `grid`。

---

## 9. 已知约束与常见问题

**数据层**

- `test.index` 是纯文本，不是 pickle，不能用 `pickle.load`；
- PyG 下载损坏（EOFError / 空文件）：自动清缓存重试一次；国内网络建议手动下载到 `data/<ClassName>/<Name>/raw/`；
- PyG 数据集名大小写敏感：`Cora` / `CiteSeer`（中间大写 S）/ `PubMed`。

**配置层**

- YAML 中 `5e-4` 会被解析为字符串，写成 `0.0005`；
- `save.monitor` 与产出指标名不匹配 → `best.pt` 从不保存 → `FileNotFoundError`；
- `save.monitor` / `save.mode` 不要放进 `GRID`。

**模型层**

- buffer 初值是空张量，`self._cached_x is not None` 恒为真，判空用 `numel() == 0`；
- `a = x if cond else y` 不会写回 `self.xxx`，缓存必须用 `if` 显式赋值；
- `register_buffer` 只接受张量，dict/list 用普通属性；
- `torch.tensor(x)` 会断开计算图，用 `.float()` / `.to()` / `.as_tensor()`；
- long @ long 不走 BLAS/cuBLAS，矩阵乘前 `.float()`；
- 稀疏图避免稠密 `(N,N) @ (N,N)`，用 `torch.sparse.mm`；
- `prepare_input` 必须写回 `self._cached_x`，否则每轮重算；
- 边计数等预处理不要放在 `forward` 里，用 buffer 缓存。

**损失 / 评估层**

- `out[mask]` 对 dict 会报 `KeyError`：返回 dict 时必须提供 `model.evaluate`，或先取出 `logits` 再索引；
- 默认 `evaluate` 只适合分类（`(pred, target)` 签名 + 有 `train_mask`），非分类任务必须自定义；
- `modularity(edge_index, labels=None)`：第一个参数永远是 `edge_index`，不要传 B 矩阵。

**训练层**

- `Trainer.fit` 中 train split 同样走 `_evaluate_split`，不要硬编码默认 evaluate；
- val / test 必须返回 monitor 对应的 key（如 `Qp` 需包成 `val_Qp`）。

---

## 10. 后续方向

- `Trainer.fit` 在 epoch 1 增加 monitor 存在性检查，失败即报错；
- 统一所有损失签名为 `(outputs, data=None)`，移除 inspect 分派；
- 抽出 `src/engine/callbacks.py`（EarlyStopping / LRScheduler / ModelCheckpoint）；
- 新增 `src/utils/io.py` 统一 yaml / json / pickle 读写；
- 网格搜索支持 `skip_existing`（断点续跑）与 `multi_seed`（多随机种子平均）；
- Amazon 系数据集加载不稳定，补充手动下载指引或替换数据源。