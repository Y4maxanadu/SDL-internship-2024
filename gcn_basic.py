import math
import random

import torch
import torch.nn.functional as Functional
import torch_sparse
from torch import nn, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import add_self_loops, degree
import dataloader as dl

AGG = 'add'
print('AGG:', AGG)


class LightGCN(MessagePassing):
    def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=64,
            K_layer=3,
            add_self_loops=False,
            init_method=1):
        super().__init__(aggr=AGG)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        self.init_method = init_method
        if self.init_method == 1:
            nn.init.normal_(self.users_emb.weight, std=0.1)
            nn.init.normal_(self.items_emb.weight, std=0.1)
        elif self.init_method == 2:
            nn.init.xavier_normal_(self.users_emb.weight)
            nn.init.xavier_normal_(self.items_emb.weight)
        elif self.init_method == 3:
            nn.init.kaiming_normal_(self.users_emb.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.items_emb.weight, mode='fan_out', nonlinearity='relu')

        print("add_self_loops", add_self_loops)
        print("init_method", init_method)

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        """
        gcn_norm
        A_hat= = ( D_hat ) ^ ( -1/2 ) * (A + I) * ( D_hat ) ^ ( 1/2 )
        where ( D_hat )_ii = Sigma_( j = 0 ) ( A_hat )_ij + 1

        This is the process of calculating the symmetrical normalized adjacency matrix
        """
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]
        # print(edge_index_norm)
        for i in range(self.K_layer):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        # split into  e_u^K and e_i^K
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

    def parameters_norm(self):
        return torch.tensor(0)


class LightAttention(LightGCN):

    def __init__(self, num_users, num_items):
        super().__init__(num_users, num_items)

        self.attention_weights = torch.nn.Parameter(
            torch.randn(self.K_layer + 1),
            requires_grad=True
        )

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]
        # print(edge_index_norm)
        for i in range(self.K_layer):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)

        # attention
        attention_scores = Functional.softmax(self.attention_weights, dim=0)
        emb_final = torch.sum(embs * attention_scores.view(1, -1, 1), dim=1)
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def parameters_norm(self):
        """
        Compute the norm of the model"s parameters.

        Returns:
            float: The norm of the model"s parameters.
        """
        return super().parameters_norm() + \
               self.attention_weights.norm(2).pow(2)


class LightScaledDotProductAttention(LightGCN):
    """
        Light Graph Convolutional Network (LightGCN) model with scaled dot-product
        attention.

    """

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]
        # print(edge_index_norm)
        for i in range(self.K_layer):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)

        # what attention
        queries, keys, values = self.prepare_attention_inputs(embs)
        attention_output = self.compute_attention(queries, keys, values)
        emb_final = torch.mean(attention_output, dim=1)
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def prepare_attention_inputs(self, embs):
        return embs, embs, embs

    @staticmethod
    def compute_attention(queries, keys, values):
        """
            Compute the scaled dot-product attention.

            Args:
                queries (torch.Tensor): Queries tensor.
                keys (torch.Tensor): Keys tensor.
                values (torch.Tensor): Values tensor.

            Returns:
                torch.Tensor: The attention output tensor.
        """
        scaling_factor = math.sqrt(queries.size(-1))
        attention_scores = torch.matmul(
            queries, keys.transpose(-2, -1)) / scaling_factor
        attention_weights = Functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output


class LightWeightedScaledDotProductAttention(LightScaledDotProductAttention):

    def __init__(self, num_users, num_items):
        """
        Initialize the WeightedScaledDotProductAttentionLightGCN model.

        """
        super().__init__(num_users, num_items)
        self.attention_dim = 1

        self.query_projection = nn.Linear(
            self.embedding_dim, self.attention_dim, bias=False)
        self.key_projection = nn.Linear(
            self.embedding_dim, self.attention_dim, bias=False)
        self.value_projection = nn.Linear(
            self.embedding_dim, self.attention_dim, bias=False)

    def prepare_attention_inputs(self, embs):
        """
        Prepare inputs for the attention mechanism.

        Args:
            embs (torch.Tensor): Embeddings tensor.

        Returns:
            tuple: A tuple of queries, keys, and values.
        """
        queries = self.query_projection(embs)
        keys = self.key_projection(embs)
        values = self.value_projection(embs)
        return queries, keys, values

    def parameters_norm(self):
        """
        Compute the norm of the model"s parameters.

        Returns:
            float: The norm of the model"s parameters.
        """
        return super().parameters_norm() + \
               self.query_projection.weight.norm(2) + \
               self.key_projection.weight.norm(2) + \
               self.value_projection.weight.norm(2)


class KOBE(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K_layer=3, add_self_loops=False, dropout_rate=0.2):
        super().__init__(aggr=None)  
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        # 初始化嵌入
        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

        # 注意力参数
        self.attention = nn.Parameter(torch.Tensor(1, 2 * self.embedding_dim))
        nn.init.xavier_normal_(self.attention)

        self.attention_weights = torch.nn.Parameter(
            torch.randn(self.K_layer + 1),
            requires_grad=True
        )
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]

        for i in range(self.K_layer):
            emb_k = self.dropout(emb_k)
            emb_k = self.propagate(edge_index_norm, x=emb_k, size=(emb_k.size(0), emb_k.size(0)))
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)

        # attention
        attention_scores = Functional.softmax(self.attention_weights, dim=0)
        emb_final = torch.sum(embs * attention_scores.view(1, -1, 1), dim=1)

        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, edge_index, x_i, x_j):
        # 计算注意力得分
        x_cat = torch.cat([x_i, x_j], dim=-1)
        attention_coefficients = Functional.leaky_relu(torch.matmul(x_cat, self.attention.T))

        return x_j * attention_coefficients

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

    def parameters_norm(self):
        return self.attention_weights.norm(2).pow(2)


class LGC_MultiHead_Attention(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K_layer=3, add_self_loops=False, heads=4):
        super().__init__(aggr=AGG)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K_layer = K_layer
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        self.attentions = nn.Parameter(torch.Tensor(self.heads, 2 * self.embedding_dim, 1))
        nn.init.xavier_normal_(self.attentions)

    def message(self, edge_index, x_i, x_j):
        x_cat = torch.cat([x_i, x_j], dim=-1)
        attention_scores = Functional.leaky_relu(x_cat @ self.attentions)
        attention_scores = torch.softmax(attention_scores.sum(dim=-1), dim=-1)
        return x_j * attention_scores.expand_as(x_j)

    def forward(self, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embs = [emb_0]

        for i in range(self.K_layer):
            emb_k = self.propagate(edge_index_norm, x=emb_k, size=(emb_k.size(0), emb_k.size(0)))
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

    def parameters_norm(self):
        """
        Returns the norm of the parameters in the model.
        """
        return torch.tensor(0)


def sample_mini_batch(batch_size, edge_index):
    """
    Randomly samples indices of a minibatch given an adjacency matrix
    :param batch_size: (int) mini batch size
    :param edge_index:  (torch.Tensor) 2 by N list of edges
    :return:tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


def bpr_loss(users_emb_k, users_emb_0, pos_item_emb_k, pos_item_emb_0, neg_item_emb_k, neg_item_emb_0, lambda_val):
    """

    :param users_emb_k:
    :param users_emb_0:
    :param pos_item_emb_k:
    :param pos_item_emb_0:
    :param neg_item_emb_k:
    :param neg_item_emb_0:
    :param lambda_val:
    :return:
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) + pos_item_emb_0.norm(2).pow(2) + neg_item_emb_0.norm(2).pow(2))

    # predicted scores of positive and negative samples
    pos_scores = torch.mul(users_emb_k, pos_item_emb_k)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_k, neg_item_emb_k)
    neg_scores = torch.sum(neg_scores, dim=-1)
    loss = - torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


def get_user_positive_items(edge_index):
    """

    :param edge_index:
    :return:
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def get_user_negative_items(edge_index):
    """

    :param edge_index:
    :return:
    """
    user_neg_items = {}
    """
    to be completed 
    """
    return user_neg_items


def recall_precision_at_K_r(ground_truth, r, k):
    """

    :param ground_truth: (list) of lists containing highly rated items
    :param r: (list) of lists indicating whether each top k item recommended to each user is a top k ground truth item or not
    :param k: top k
    :return:
    """
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor([len(ground_truth[i]) for i in range(len(ground_truth))])
    recall = torch.mean(num_correct_pred / user_num_liked).item()
    precision = torch.mean(num_correct_pred) / k
    return recall, precision.item()


def NDCGat_K_r(ground_truth, r, k):
    """

    :param groundTruth:
    :param r:
    :param k:
    :return:
    """
    assert len(r) == len(ground_truth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(ground_truth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r / torch.log2(torch.arange(2, k + 2)), dim=1)
    dcg = r / torch.log2(torch.arange(2, k + 2))
    dcg = torch.sum(dcg, dim=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


class NGCF(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K_layer=3, add_self_loops=False, dropout_rate=0.2):
        super(NGCF, self).__init__(aggr='add')
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

        # Define transformation matrices for each layer
        self.W1 = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(K_layer)])
        self.W2 = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(K_layer)])

    def forward(self, edge_index: SparseTensor):
        if self.add_self_loops:
            edge_index = edge_index.set_diag()

        row, col, edge_weight = edge_index.coo()

        if edge_weight is None:
            edge_weight = torch.ones_like(row, dtype=torch.float)

        deg = degree(row, self.num_users + self.num_items, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight_norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0
        embeddings = [emb_0]

        for i in range(self.K_layer):
            emb_k = self.dropout(emb_k)
            emb_k = self.propagate(edge_index=edge_index, edge_weight=edge_weight_norm, x=emb_k, W1=self.W1[i],
                                   W2=self.W2[i])
            embeddings.append(emb_k)

        embs = torch.stack(embeddings, dim=1)
        emb_final = torch.mean(embs, dim=1)

        users_emb_k, items_emb_k = torch.split(emb_final, [self.num_users, self.num_items])

        return users_emb_k, self.users_emb.weight, items_emb_k, self.items_emb.weight

    def message(self, x_i, x_j, edge_weight, W1, W2):
        return torch.relu(W1(x_i) + W2(x_j * edge_weight.view(-1, 1)))

    def message_and_aggregate(self, adj_t: SparseTensor, x: torch.Tensor) -> torch.Tensor:
        return matmul(adj_t, x)

    def parameters_norm(self):
        return torch.tensor(0)

class GAT(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, heads=1, dropout_rate=0.2, add_self_loops=False):
        super(GAT, self).__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.dropout_rate = dropout_rate

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        self.att = nn.Parameter(torch.Tensor(1, heads, embedding_dim * 2))
        nn.init.xavier_normal_(self.att)

        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, edge_index: SparseTensor):
        if self.add_self_loops:
            edge_index = edge_index.set_diag()

        row, col, edge_weight = edge_index.coo()

        if edge_weight is None:
            edge_weight = torch.ones_like(row, dtype=torch.float)

        edge_index_norm = torch.stack([row, col], dim=0)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        emb_k = emb_0

        emb_k = self.dropout(emb_k)
        alpha = self.edge_attention(emb_k, row, col)

        emb_k = self.propagate(edge_index=edge_index_norm, x=emb_k, alpha=alpha, edge_weight=edge_weight)

        users_emb_k, items_emb_k = torch.split(emb_k, [self.num_users, self.num_items])
        return users_emb_k, self.users_emb.weight, items_emb_k, self.items_emb.weight

    def edge_attention(self, x, row, col):
        x_i = x[row]
        x_j = x[col]
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = self.leaky_relu((alpha * self.att).sum(dim=-1))
        return alpha

    def message(self, x_j, alpha):
        alpha = torch.exp(alpha)
        return x_j * alpha.view(-1, 1)

    def aggregate(self, inputs, index):
        alpha_sum = torch.zeros((index.max() + 1, inputs.size(-1)), device=inputs.device)
        alpha_sum = alpha_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(inputs), inputs)
        alpha_sum[alpha_sum == 0] = 1e-9
        return inputs / alpha_sum[index].unsqueeze(1)

    def update(self, aggr_out):
        return aggr_out