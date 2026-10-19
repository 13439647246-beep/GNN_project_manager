# runs/ — 训练产物输出目录

本目录由训练引擎自动生成，**不需要手动创建或管理**。所有产物按 **模型 → 运行时间戳 → 数据集 → 实验类型（单次/网格）** 组织。

> ⚠️ 本目录整体不纳入 git 版本控制（见根目录 `.gitignore`）。需要长期保留的实验请自行打包归档。

---

## 一、目录结构

```text
runs/
└── <model_name>/                   # 模型名，如 HGCNRH / GCN / KPGN
    └── <timestamp>/                # 运行时间戳，格式：YYYY-MM-DD_HH-MM-SS
        └── <dataset_name>/         # 数据集名，如 cora / citeseer / pubmed
            ├── <exp_name>/         # [单次模式] 若为单次运行，这里直接是实验目录
            │
            └── _grid/              # [网格模式] 网格搜索的专属目录
                ├── combo1/         # 第 1 个参数组合
                │   ├── ckpt/
                │   │   ├── best.pt
                │   │   └── last.pt
                │   ├── config.yaml # ★ 该组合的完整参数快照
                │   ├── metrics.csv
                │   └── train.log
                ├── combo2/         # 第 2 个参数组合
                │   └── ...
                ├── combo3/         # 第 3 个参数组合
                └── summary.csv     # (可选) 若代码生成，则存放 combo 对照表
```

**层级含义**：

| 层级           | 说明                                                       | 示例                  |
|----------------|------------------------------------------------------------|-----------------------|
| `model_name`   | 对应 `models/<name>/` 和 `configs/<name>.yaml`             | `HGCNRH` / `GCN`      |
| `timestamp`    | 本次运行启动时间，一次运行（多数据集）共用一个             | `2026-10-19_10-39-40` |
| `dataset_name` | 对应 `src/data/` 注册的名字                                | `cora` / `pubmed`     |
| `_grid`        | 网格搜索标记，只有调参时才会出现这个文件夹                 | `_grid`               |
| `combo<i>`     | 第 i 个参数组合，目录名精简，避免超过 Windows 260 字符限制 | `combo1`              |

---

## 二、为什么用 `_grid/combo<i>` 组织？

1. **隔离单次与网格**：看目录里有 `_grid`，就知道跑过调参；没有 `_grid`，就是单次对比实验。
2. **避免文件名过长**：参数一多（`lr=0.001_dropout=0.5_hidden=256...`），不仅难读，在 Windows 下极易触发路径超长报错。`combo1` 干净利落。
3. **便于批量清理**：如果不满意这轮调参，直接 `rm -rf _grid` 即可删掉所有组合，保留其他实验。

---

## 三、如何查"combo1 对应什么参数"

因为目录名只有 `combo1`，**参数详情必须去文件里找**：

### 方式 1：看 `config.yaml`（最准）
```bash
cat runs/HGCNRH/2026-10-19_10-39-40/cora/_grid/combo1/config.yaml
```
里面包含该组合被注入的所有超参（lr、dropout、lambda 等）。

### 方式 2：看 `summary.csv`（如果你在代码里实现了）
`_grid/summary.csv` 会列出所有 combo 的得分和参数映射。

---

## 四、单次模式 vs 网格模式对比

### 单次模式（不调参）
```text
runs/
└── HGCNRH/
    └── 2026-10-19_10-39-40/
        └── cora/
            └── HGCNRH_run/          # 直接建实验目录
                ├── ckpt/
                ├── config.yaml
                ├── metrics.csv
                └── train.log
```

### 网格模式（截图中的结构）
```text
runs/
└── HGCNRH/
    └── 2026-10-19_10-39-40/
        └── cora/
            └── _grid/               # ★ 统一收在 _grid 下
                ├── combo1/
                ├── combo2/
                └── combo3/
```

---

## 五、代码实现对照（需要与你的代码逻辑一致）

### 1. `src/engine/logger.py` —— 路径拼接逻辑
```python
# 获取当前运行时间戳（全项目共用）
ts = arg["exp"].get("timestamp") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 判断是否为网格模式
is_grid = arg["exp"]["name"].startswith("combo")

if is_grid:
    # 网格模式：数据名/_grid/comboX
    self.run_dir = os.path.join(
        arg["exp"]["output_dir"],   # "runs"
        arg["model"]["name"],       # "HGCNRH"
        ts,                         # "2026-10-19_10-39-40"
        arg["data"]["name"],        # "cora"
        "_grid",                    # ★ 固定文件夹
        arg["exp"]["name"],         # "combo1"
    )
else:
    # 单次模式：数据名/实验名
    self.run_dir = os.path.join(
        arg["exp"]["output_dir"],
        arg["model"]["name"],
        ts,
        arg["data"]["name"],
        arg["exp"]["name"],         # "HGCNRH_run"
    )

os.makedirs(self.run_dir, exist_ok=True)
```

### 2. `src/config/grid.py` —— combo 编号生成
```python
def expand_grid(base_arg, grid):
    if not grid:
        yield _with_exp_name(copy.deepcopy(base_arg), applied={}, index=1)
        return

    keys = list(grid.keys())
    values = [list(v) if isinstance(v, (list, tuple)) else [v] for v in grid.values()]

    for i, combo in enumerate(itertools.product(*values), start=1):
        arg = copy.deepcopy(base_arg)
        applied = {}
        for k, v in zip(keys, combo):
            set_nested(arg, k, v)
            applied[k] = v
        
        # ★ 重命名为 combo1, combo2...
        arg["exp"]["name"] = f"combo{i}"
        arg["exp"]["params"] = dict(applied)  # 把参数存下来，供 summary.csv 用
        yield arg
```

### 3. `src/engine/runner.py` —— summary 存放位置
```python
def save_summary(results, base_arg):
    # 汇总文件放在 _grid 根目录下
    out_dir = os.path.join(
        base_arg["exp"]["output_dir"],
        base_arg["model"]["name"],
        base_arg["exp"]["timestamp"],
        base_arg["data"]["name"],
        "_grid",
    )
    os.makedirs(out_dir, exist_ok=True)
    
    path = os.path.join(out_dir, "summary.csv")
    # ... 写入逻辑（加上 params 列）...
```

### 4. `run_test.py` —— 保证时间戳唯一
```python
from datetime import datetime

# 在进入 main 的第一时间生成，确保多数据集共用
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    arg = load_arg(...)
    arg["exp"]["timestamp"] = RUN_TIMESTAMP
    
    for data_name in DATA_NAMES:
        arg["data"]["name"] = data_name
        # ... 后续流程 ...
```

---

## 六、常见操作

**只保留 best combo，删掉其他（清理空间）**：
```bash
cd runs/HGCNRH/2026-10-19_10-39-40/cora/_grid/
# 假设 combo3 最好
ls -d combo* | grep -v combo3 | xargs rm -rf
```

**对比不同数据集的同参数表现**：
因为 `combo1` 在不同数据集下对应的是同一组超参，所以可以直接比较：
```python
# cora 的 combo1 vs pubmed 的 combo1
path_cora = "runs/HGCNRH/.../cora/_grid/combo1/train.log"
path_pubmed = "runs/HGCNRH/.../pubmed/_grid/combo1/train.log"
```

**读取参数对照表**（如果你写了 `summary.csv`）：
```python
import pandas as pd
df = pd.read_csv("runs/HGCNRH/2026-10-19_10-39-40/cora/_grid/summary.csv")
print(df.sort_values("val_Qp", ascending=False))
```