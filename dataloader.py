from time import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_sparse import SparseTensor


dataset_list = [
    'resume',
    'movie',
    'fake_resume',
    'gowalla'
]


def select_dataset(_):
    return dataset_list[int(_)]


folder_name = select_dataset(input('which dataset:'))


def get_folder_name():
    return folder_name


item_path = 'data/' + folder_name + '/item.csv'
rating_path = 'data/' + folder_name + '/rating.csv'
print('we are processing data from: ' + rating_path)

MINIMUM_RATING = 1
print('we are using rating threshold', MINIMUM_RATING)


def load_node_csv(path, index_col):
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold):
    df = pd.read_csv(path)

    src, dst = [src_mapping[index] for index in df[src_index_col]], [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold
    # print('edge_attr', edge_attr, 'src', src, 'dst', dst)
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    # print("there are", len(edge_index[0]), 'edges')
    return torch.tensor(edge_index)


def get_user_mapping():
    return load_node_csv(rating_path, index_col='user_id')


def get_resume_mapping():
    return load_node_csv(item_path, index_col='resume_id')


def get_num_users(user_mapping):
    return len(user_mapping)


def get_num_resumes(resume_mapping):
    return len(resume_mapping)


def get_edge_index():
    edge_index = load_edge_csv(
        rating_path,
        src_mapping=get_user_mapping(),
        src_index_col='user_id',
        dst_mapping=get_resume_mapping(),
        dst_index_col='resume_id',
        link_index_col='rating',
        rating_threshold=MINIMUM_RATING
    )
    return edge_index


def get_sparse_edge_index(n1, n2):
    edge_index = get_edge_index()
    num_interactions = edge_index.shape[1]

    all_indices = [i for i in range(num_interactions)]
    train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=None)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=None)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                           sparse_sizes=(n1 + n2, n1 + n2))
    val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1],
                                         sparse_sizes=(n1 + n2, n1 + n2))
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                          sparse_sizes=(n1 + n2, n1 + n2))

    return [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index, train_edge_index, val_edge_index,
            test_edge_index]


def getSparseGraph(num_user, num_item):
    print("generating adjacency matrix")
    s = time()
    adj_mat = sp.dok_matrix((num_user + num_item, num_user + num_item), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    df = pd.read_csv('data/resume/rating.csv')
    train_user = df["user_id"].to_numpy() - 1
    train_item = df["resume_id"].to_numpy() - 1
    user_item_net = csr_matrix((np.ones(len(train_user)),
                                (train_user, train_item)),
                               shape=(num_user, num_item))
    R = user_item_net.tolil()
    adj_mat[:num_user, num_user:] = R
    adj_mat[num_user:, :num_user] = R.T
    adj_mat = adj_mat.todok()

    row_sum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    end = time()
    print(f"costing {end - s}s, saved norm_mat...")
    sp.save_npz('/data/matrix/s_pre_adj_mat.npz', norm_adj)

    graph = _convert_sp_mat_to_sp_tensor(norm_adj)
    graph = graph.coalesce().to('cpu')

    return graph


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


"""

def load_edge_csv1(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=6):
    #print('we are using rating threshold', rating_threshold)

    df = pd.read_csv(path)
    src, dst = [src_mapping[index] for index in df[src_index_col]], [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >4
    edge_attr1 = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >8

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    #print("there are", len(edge_index[0]), 'edges')

    edge_index1 = [[], []]
    for i in range(edge_attr1.shape[0]):
        if edge_attr1[i]:
            edge_index1[0].append(src[i])
            edge_index1[1].append(dst[i])
    #print("there are", len(edge_index1[0]), '>8 edges')
    return torch.tensor(edge_index), torch.tensor(edge_index1)


def load_dynamic_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col,
                          rating_threshold=6):
    print('we are using rating threshold', rating_threshold)
    df = pd.read_csv(path)
    src, dst = [src_mapping[index] for index in df[src_index_col]], [dst_mapping[index] for index in df[dst_index_col]]

    edge_attrs = [[] for _ in range(5)]

    for i in range(5):
        edge_attrs[i].append(torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) == 6 + i)
    edge_indices = [[[], []], [[], []], [[], []], [[], []], [[], []]]
    tensor_edge_index = []

    print(edge_attrs)

    for k in range(5):
        for i in range(len(edge_attrs[k][0])):
            if edge_attrs[k][0][i]:
                # print(src[i], dst[i])
                edge_indices[k][0].append(src[i])
                edge_indices[k][1].append(dst[i])

        print("there are", len(edge_indices[k][0]), k + 6, 'value edges')
        tensor_edge_index.append(torch.tensor(edge_indices[k]))

    return tensor_edge_index

def get_edge_index1():
    edge_index, edge_index1 = load_edge_csv1(
        rating_path,
        src_mapping=get_user_mapping(),
        src_index_col='user_id',
        dst_mapping=get_resume_mapping(),
        dst_index_col='resume_id',
        link_index_col='rating',
        rating_threshold=6
    )
    return edge_index, edge_index1


def get_dynamic_edge_index():
    edge_index_p = load_dynamic_edge_csv(
        rating_path,
        src_mapping=get_user_mapping(),
        src_index_col='user_id',
        dst_mapping=get_resume_mapping(),
        dst_index_col='resume_id',
        link_index_col='rating',
        rating_threshold=6
    )
    return edge_index_p


def get_sparse_edge_index1(n1, n2):
    edge_index, edge_index1 = get_edge_index1()


    #print(edge_index)
    #print(edge_index1)


    num_interactions = edge_index.shape[1]
    num_interactions1 = edge_index1.shape[1]
    all_indices = [i for i in range(num_interactions)]
    train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=None)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=None)

    all_indices1 = [i for i in range(num_interactions1)]
    train_indices1, test_indices1 = train_test_split(all_indices1, test_size=0.3, random_state=None)
    val_indices1, test_indices1 = train_test_split(test_indices1, test_size=0.5, random_state=None)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    train_edge_index1 = edge_index1[:, train_indices1]
    val_edge_index1 = edge_index1[:, val_indices1]
    test_edge_index1 = edge_index1[:, test_indices1]

    # print("Train edge Index", train_edge_index)

    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                           sparse_sizes=(n1 + n2, n1 + n2))
    val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1],
                                         sparse_sizes=(n1 + n2, n1 + n2))
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                          sparse_sizes=(n1 + n2, n1 + n2))

    train_sparse_edge_index1 = SparseTensor(row=train_edge_index1[0], col=train_edge_index1[1],
                                            sparse_sizes=(n1 + n2, n1 + n2))
    val_sparse_edge_index1 = SparseTensor(row=val_edge_index1[0], col=val_edge_index1[1],
                                          sparse_sizes=(n1 + n2, n1 + n2))
    test_sparse_edge_index1 = SparseTensor(row=test_edge_index1[0], col=test_edge_index1[1],
                                           sparse_sizes=(n1 + n2, n1 + n2))
    # print(test_sparse_edge_index)

    return [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index, train_edge_index, val_edge_index,
            test_edge_index], [train_sparse_edge_index1, val_sparse_edge_index1, test_sparse_edge_index1,
                               train_edge_index1, val_edge_index1,
                               test_edge_index1]


def get_sparse_dynamic_edge_index(n1, n2):
    edge_indices = []
    edge_indices = get_dynamic_edge_index()
    num_interactions = []
    for i in range(5):
        num_interactions.append(edge_indices[i].shape[1])

    all_indices = [[] for _ in range(5)]

    train_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]

    train_edge_index = [[] for _ in range(5)]
    test_edge_index = [[] for _ in range(5)]
    val_edge_index = [[] for _ in range(5)]

    train_sparse_edge_index = [[] for _ in range(5)]
    test_sparse_edge_index = [[] for _ in range(5)]
    val_sparse_edge_index = [[] for _ in range(5)]

    for i in range(5):
        temp = [j for j in range(num_interactions[i])]
        # all_indices[i].append(temp)
        all_indices[i] = temp
        train_indices[i], test_indices[i] = train_test_split(all_indices[i], test_size=0.2, random_state=None)
        val_indices[i], test_indices[i] = train_test_split(test_indices[i], test_size=0.5, random_state=None)

        train_edge_index[i] = edge_indices[i][:, train_indices[i]]
        val_edge_index[i] = edge_indices[i][:, val_indices[i]]
        test_edge_index[i] = edge_indices[i][:, test_indices[i]]

        train_sparse_edge_index[i] = SparseTensor(row=train_edge_index[i][0], col=train_edge_index[i][1],
                                                  sparse_sizes=(n1 + n2, n1 + n2))
        test_sparse_edge_index[i] = SparseTensor(row=test_edge_index[i][0], col=test_edge_index[i][1],
                                                 sparse_sizes=(n1 + n2, n1 + n2))
        val_sparse_edge_index[i] = SparseTensor(row=val_edge_index[i][0], col=val_edge_index[i][1],
                                                sparse_sizes=(n1 + n2, n1 + n2))

    # print(val_sparse_edge_index)

    return [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index, train_edge_index, val_edge_index,
            test_edge_index]

"""
