import torch
from torch import nn, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul


class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K_layer=6, add_self_loops=False, dropout_rate=0.2):
        super(LightGCN, self).__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        self.dropout_rate = dropout_rate

        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embeddings = [emb_0]

        for i in range(self.K_layer):
            emb_k = self.dropout(emb_k)
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embeddings.append(emb_k)

        embs = torch.stack(embeddings, dim=1)
        emb_final = torch.mean(embs, dim=1)

        # split into  e_u^K and e_i^K
        users_emb_k, items_emb_k = torch.split(emb_final, [self.num_users, self.num_items])

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_k, self.users_emb.weight, items_emb_k, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

    def parameters_norm(self):
        return torch.tensor(0)


class IMP_GCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K_layer=6, add_self_loops=False, dropout_rate=0.2,
                 groups=3):
        super(IMP_GCN, self).__init__(aggr='add')  # aggregation method as 'add'
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops
        self.groups = groups
        self.dropout_rate = dropout_rate

        # Embeddings for users and items
        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)

        # Group-specific transformations
        self.fc_g = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(groups)])

        # Initialize weights
        nn.init.xavier_uniform_(self.users_emb.weight)
        nn.init.xavier_uniform_(self.items_emb.weight)
        for fc in self.fc_g:
            nn.init.xavier_uniform_(fc.weight)

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, edge_index: SparseTensor):
        # Normalize edge indices
        edge_index = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        # Get initial embeddings
        user_embeddings = self.users_emb.weight
        item_embeddings = self.items_emb.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings])

        # List to store embeddings from each layer
        all_emb_list = [all_embeddings]

        for _ in range(self.K_layer):
            all_embeddings = self.propagate(edge_index, x=all_embeddings)
            all_emb_list.append(all_embeddings)

        # Aggregate embeddings from all layers
        all_embeddings = torch.stack(all_emb_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        # Split embeddings back into users and items
        users_emb_final, items_emb_final = torch.split(all_embeddings, [self.num_users, self.num_items])

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: torch.Tensor, norm) -> torch.Tensor:
        # Message function used by 'propagate'. Optionally uses 'norm' if it is not None.
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: torch.Tensor) -> torch.Tensor:
        return matmul(adj_t, x)

    def parameters_norm(self):
        return torch.tensor(0)