import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional
import torch


class NeighborAggregatorLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点的邻居（只是对邻居做处理）

        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregatorLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        # neighbor_feature: [N, t, input_dim]
        neighbor_hidden = torch.matmul(neighbor_feature, self.weight)  # [N, t, output_dim]
        if self.use_bias:
            neighbor_hidden += self.bias

        # 在邻居维度（dim=1）上聚合
        if self.aggr_method == "mean":
            neighbor_hidden = neighbor_hidden.mean(dim=1)
        elif self.aggr_method == "sum":
            neighbor_hidden = neighbor_hidden.sum(dim=1)
        elif self.aggr_method == "max":
            neighbor_hidden = neighbor_hidden.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported aggr_method: {self.aggr_method}")

        # 现在 neighbor_hidden 形状是 [N, output_dim]
        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class SageGCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=functional.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN层定义

        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCNLayer, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregatorLayer(input_dim, hidden_dim,
                                                  aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        # node_fea * w ; nei_fea
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSageLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_neighbors_list):
        super(GraphSageLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCNLayer(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 1):
            self.gcn.append(SageGCNLayer(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(SageGCNLayer(hidden_dim[-1], output_dim, activation=None))
        # gcn就是sageGCN+relu的叠加
    def forward(self, node_features_list):
        # node_features_list：一个list 第0个元素表示源节点特征，其后的表示每阶采样得来的特征
        hidden = node_features_list
        # 对num_neighbors_list中的每一个
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                # 第0层节点们的隐状态
                src_node_num = len(src_node_features)
                # 第0层节点们的个数
                neighbor_node_features = hidden[hop + 1] .view((src_node_num, self.num_neighbors_list[hop], -1))
                # neighbor_node_features 是由[第0层节点的个数*第0层节点邻居的个数*自动长度]的向量
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )

class KPGNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_neighbors_list, num_classes):
        super(KPGNLayer, self).__init__()
        self.graphsage = GraphSageLayer(input_dim, hidden_dims, output_dim, num_neighbors_list)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, node_features_list):
        embeddings = self.graphsage(node_features_list)
        logits = self.classifier(embeddings)
        return embeddings, logits