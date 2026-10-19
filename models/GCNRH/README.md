Graph convolutional network reconstruction with high-order node information for community detection
https://doi.org/10.1016/j.engappai.2025.112957
Highlights
• GCNRH reveals the network community structure by reconstructing the modularity matrix.
• GCNRH uses the biaffine attention mechanism to capture nodes’ high-order neighbor information.
• Comparative experiments show that GCNRH performs excellently on 12 public datasets.

文章的主要贡献是构建了高阶节点信息图卷积网络重建模型（GCNRH），采用双仿射注意力模型在不增加GCN深度的前提下对图中所有节点及其连接进行分析。
文章以核心基于模块度矩阵的重构进行，故首先计算网络中模块度矩阵并与节点特征拼接并进行归一化。一方面将特征矩阵与节点邻接矩阵一起输入给d维的GCN提取邻域特征，另一方面输入d维全连接模型获得自节点特征。
GCN 分支得到的是局部邻域特征，它聚合了一阶邻居信息；全连接分支得到的是自节点特征表示，保留节点自身原始特征。使用全连接层是为了避免完全依赖邻域聚合导致的“特征退化”或“过度平滑”。
得到的特征矩阵通过双仿射注意力模型分别将邻居特征和自节点特征引入对方，并以此更新特征矩阵。不依赖图的边，直接计算所有节点两两之间的注意力，从而在不堆叠 GCN 层的情况下捕获高阶、长距离信息。
之后通过池化层把原始特征和双仿射更新后的特征逐元素相加并进行归一化分别描述GCN领域聚合（均值μ）和个体特征（标准差β）。最终解码过程构建𝐶 = 𝜇 + 𝛽𝐑描述每个节点的连接特征和个体特征，其中R是正态分布噪声矩阵。C的内积即描述各节点与其他节点连接相似性，也即描述节点是否属于同一个社区。模型联合使用模块度矩阵的交叉熵和KL散度正则化损失作为损失函数实现反向传播。
缺陷： 模块度矩阵与节点特征简单拼接，可能损失信息；没有展开说如何构建了社区，只是重构了模块度矩阵；