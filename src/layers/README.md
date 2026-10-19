\# src/layers



跨模型可复用的层与块。



\## 文件职责



| 文件 | 内容 |

|---|---|

| `basic.py` | 激活工厂、MLP、残差块 |

| `graph.py` | 图卷积块的通用封装 + GCN/GAT/SAGE/GIN 专用块 |

| `attention.py` | 多头自注意力 |

| `initialization.py` | 权重初始化方案 |



\## 上提 vs 下沉的判断



\- \*\*跨 2 个以上模型用到\*\* → 上提到这里

\- \*\*只在一个模型里用\*\* → 留在 `models/<name>/layer.py`

\- \*\*PyG 已提供原子层\*\* → 只做组合封装，不重复实现



\## 使用示例



```python

from src.layers import MLP, make\_gcn\_block, apply\_init



class MyModel(nn.Module):

&#x20;   def \_\_init\_\_(self, arg):

&#x20;       super().\_\_init\_\_()

&#x20;       self.block1 = make\_gcn\_block(1433, 16, dropout=0.5)

&#x20;       self.block2 = make\_gcn\_block(16, 7, activation=None)

&#x20;       self.head   = MLP(7, 7, 7, num\_layers=1)

&#x20;       apply\_init(self, scheme="xavier")



&#x20;   def forward(self, data):

&#x20;       x = data.x

&#x20;       x = self.block1(x, data.edge\_index)

&#x20;       x = self.block2(x, data.edge\_index)

&#x20;       return self.head(x)

