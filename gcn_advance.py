import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul


class KOBE(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K_layer=3, add_self_loops=False, dropout_rate=0.2):
        super().__init__(aggr=None)  # 自定义聚合
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops
        self.dropout_rate = dropout_rate

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

        # 注意力参数
        self.attention = nn.Parameter(torch.Tensor(1, 2 * self.embedding_dim))
        nn.init.xavier_normal_(self.attention)

        # 初始化注意力权重
        self.attention_weights = nn.Parameter(torch.linspace(1.0, 0.1, self.K_layer + 1))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]

        for i in range(self.K_layer):
            emb_k = self.dropout(emb_k)
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)

        # 应用递减的注意力权重
        attention_scores = torch.softmax(self.attention_weights, dim=0)
        emb_final = torch.sum(embs * attention_scores.view(1, -1, 1), dim=1)

        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight
    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

    def parameters_norm(self):
        return torch.tensor(0)