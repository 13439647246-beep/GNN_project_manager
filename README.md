### 项目结构：

project\_root/                     # 项目根目录

│

├── src/                          # ★ 核心库（可复用代码全在这里）

│   │

│   ├── \_\_init\_\_.py               # 包标识，可选择性导出常用函数

│   │

│   ├── data/                     # ★ 通用数据管理（所有模型共享）

│   │   ├── \_\_init\_\_.py           # 导出 load\_data / DATASETS 等对外接口

│   │   ├── datasets.py           # 各公开数据集的定义与下载/读取逻辑（Cora、Citeseer、Pubmed…）

│   │   ├── loader.py             # 通用加载函数：load\_data(name, arg) 返回统一格式

│   │   ├── split.py              # 数据划分：训练/验证/测试切分、随机种子控制

│   │   └── transform.py          # 数据预处理/特征归一化/图结构处理等通用变换

│   │

│   ├── layers/                   # ★ 通用网络层（跨模型复用的基础模块）

│   │   ├── \_\_init\_\_.py           # 导出常用层，方便 from src.layers import ...

│   │   ├── basic.py              # 基础层：MLP、ConvBNReLU、Linear 封装等

│   │   ├── attention.py          # 注意力层：自注意力、多头注意力、图注意力

│   │   ├── graph.py              # 图相关层：图卷积、消息传递、邻接矩阵运算封装

│   │   └── init.py               # 权重初始化策略（xavier、kaiming、orthogonal 等）

│   │

│   ├── ops/                      # ★ 通用数理运算函数（无状态的纯函数）

│   │   ├── \_\_init\_\_.py           # 导出常用运算

│   │   ├── functional.py         # 通用函数：归一化、softmax、mask、采样等

│   │   ├── linalg.py             # 线代运算：矩阵分解、相似度、谱运算

│   │   ├── metrics.py            # 评估指标：准确率、F1、AUC、MSE 等

│   │   └── utils.py              # 杂项数值工具：one-hot、随机种子、类型转换

│   │

│   ├── engine/                   # ★ 训练引擎（所有模型共用同一套流程）

│   │   ├── \_\_init\_\_.py           # 导出 Trainer / Evaluator / Checkpoint 等

│   │   ├── trainer.py            # 训练主循环：forward → loss → backward → step

│   │   ├── evaluator.py          # 验证/测试循环：计算指标、返回结果字典

│   │   ├── checkpoint.py         # 模型保存/读取/resume 的统一接口

│   │   ├── logger.py             # 日志记录：控制台 + 文件 + CSV/TensorBoard

│   │   └── callbacks.py          # 训练回调：EarlyStopping、最佳模型保存等

│   │

│   ├── config/                   # ★ 超参管理

│   │   ├── \_\_init\_\_.py           # 导出 default\_arg / load\_arg / merge\_arg 等

│   │   ├── default\_arg.py        # 全局默认 arg 字典（所有模型读同一份结构）

│   │   ├── arg\_io.py             # arg 的读取/保存/深合并/命令行无关的覆盖逻辑

│   │   └── grid.py               # 网格展开：给定参数列表生成所有 arg 组合

│   │

│   └── utils/                    # 通用工具

│       ├── \_\_init\_\_.py           # 导出工具函数

│       ├── seed.py               # 全局随机种子固定（torch/numpy/random）

│       ├── device.py             # 设备选择：自动判断 cuda/mps/cpu

│       ├── io.py                 # 文件读写：yaml/json/pickle 存取

│       └── registry.py           # 注册表：模型/数据集/损失按名字注册，避免写 if-else

│

├── models/                       # ★ 每个模型一个独立文件夹

│   │

│   ├── \_\_init\_\_.py               # 统一 import 所有模型子包，触发注册

│   │

│   ├── gcn/                      # 例：GCN 模型

│   │   ├── \_\_init\_\_.py           # 导出 build\_model / build\_loss，并注册到 registry

│   │   ├── layer.py              # ★ 本模型特有的网络层（如 GraphConvLayer）

│   │   ├── model.py              # ★ 模型完整声明：组装层、定义 forward

│   │   ├── loss.py               # ★ 本模型的损失函数（如交叉熵 + 正则）

│   │   └── README.md             # 论文链接、复现说明、指标记录

│   │

│   ├── gat/                      # 例：GAT 模型（结构同上）

│   │   ├── \_\_init\_\_.py

│   │   ├── layer.py

│   │   ├── model.py

│   │   ├── loss.py

│   │   └── README.md

│   │

│   └── mlp/                      # 例：MLP 基线（结构同上）

│       ├── \_\_init\_\_.py

│       ├── layer.py

│       ├── model.py

│       ├── loss.py

│       └── README.md

│

├── configs/                      # ★ 超参配置文件

│   ├── \_base.yaml                # 全局默认 arg（所有模型继承的公共部分）

│   ├── gcn.yaml                  # GCN 模型的基线超参

│   ├── gat.yaml                  # GAT 模型的基线超参

│   └── grid/                     # 网格搜索配置

│       └── gcn\_grid.yaml         # GCN 的网格搜索参数列表

│

├── runs/                         # ★ 训练产物（建议 gitignore）

│   └── <model>/<exp>/<时间戳>/    # 每次运行独立目录

│       ├── config.yaml           # 本次运行的完整 arg 快照（可追溯）

│       ├── train.log             # 训练日志

│       ├── metrics.csv           # 每轮指标记录

│       └── ckpt/

│           ├── best.pt           # 最佳模型权重

│           └── last.pt           # 最后一轮权重（用于 resume）

│

├── data/                         # 原始数据集存放目录（Cora 等下载到这里）

│

├── tests/                        # 库内单元测试（可选，针对 layers/ops 做回归）

│   ├── test\_layers.py            # 测试通用层的前向/形状是否正确

│   ├── test\_ops.py               # 测试数理运算函数的正确性

│   └── test\_arg.py               # 测试 arg 加载/合并/网格展开逻辑

│

├── requirements.txt              # 依赖清单

├── pyproject.toml                # 项目元信息与打包配置

└── README.md                     # 项目说明

│

└── run\_test.py                   # ★ 项目外测试脚本（放在根目录，直接 python run\_test.py 运行）

