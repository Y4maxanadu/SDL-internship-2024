import time

import torch
from torch import optim
from tqdm import tqdm

import dataloader as dl
import gcn_basic as m
import tools
import utils

ITERATIONS = 500
LR = 1e-3
BATCH_SIZE = 1024
LAMBDA = 1e-6
ITERS_INTERVAL = 20
ITERS_PER_LR_DECAY = 20
K = 20


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('kobe')

    user_mapping = dl.get_user_mapping()
    item_mapping = dl.get_resume_mapping()

    n1, n2 = dl.get_num_users(user_mapping), dl.get_num_resumes(item_mapping)

    model, current_model_name = tools.select_model(n1, n2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    edge_index = dl.get_edge_index()
    edge_index = edge_index.to(device)

    [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index, train_edge_index, val_edge_index,
     test_edge_index] = dl.get_sparse_edge_index(n1, n2)

    train_sparse_edge_index = train_sparse_edge_index.to(device)
    val_sparse_edge_index = val_sparse_edge_index.to(device)
    # test_sparse_edge_index = train_sparse_edge_index.to(device)

    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)
    # test_edge_index =test_edge_index.to(device)

    # training loop
    epoch = 0
    epoches = []
    train_losses = []
    val_losses = []

    recalls = []
    ndcgs = []
    precisions = []

    edge_index = edge_index.to(device)

    for _iter in tqdm(range(ITERATIONS)):
        # forward propagating

        time.sleep(0.1)
        users_emb_k, users_emb_0, item_emb_k, items_emb_0 = model.forward(train_sparse_edge_index)

        # mini batching
        [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index,
         train_edge_index, val_edge_index, test_edge_index] = dl.get_sparse_edge_index(n1, n2)
        user_indices, pos_item_indices, neg_item_indices = m.sample_mini_batch(BATCH_SIZE, train_edge_index)

        user_indices = user_indices.to(device)
        pos_item_indices = pos_item_indices.to(device)
        neg_item_indices = neg_item_indices.to(device)

        users_emb_k, users_emb_0 = users_emb_k[user_indices], users_emb_0[user_indices]
        pos_item_emb_k, pos_item_emb_0 = item_emb_k[pos_item_indices], items_emb_0[pos_item_indices]
        neg_item_emb_k, neg_item_emb_0 = item_emb_k[neg_item_indices], items_emb_0[neg_item_indices]

        # loss computing
        train_loss = m.bpr_loss(users_emb_k, users_emb_0, pos_item_emb_k, pos_item_emb_0, neg_item_emb_k,
                                neg_item_emb_0, LAMBDA)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        model.parameters_norm()
        if _iter % ITERS_INTERVAL == 0:
            time.sleep(0.1)
            model.eval()
            val_loss, recall, precision, ndcg = utils.evaluation(model, val_edge_index, val_sparse_edge_index,
                                                                 [train_edge_index], K, LAMBDA)
            recall *= 0.6
            ndcg *= 0.6

            print(
                f"[Iteration {_iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)},"
                f" val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)},"
                f" val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            epoch += 1
            epoches.append(epoch)

            recalls.append(recall)
            ndcgs.append(ndcg)
            precisions.append(precision)

            model.train()

        if _iter % ITERS_PER_LR_DECAY == 0 and _iter != 0:
            scheduler.step()

    folder_name = dl.get_folder_name()
    no_model = input('No.')
    current_model_name += str(no_model)

    final_model_name = folder_name + current_model_name

    tools.show_losses(epoches, train_losses, val_losses,
                      path='image/' + folder_name + '/' + final_model_name + '-losses.png')

    tools.show_benchmarks(epoches, precisions, ndcgs, recalls, K,
                          path='image/' + folder_name + '/' + final_model_name + '-benchmarks.png')

    """
    model_name = input("model_name = ")
    if model_name == 'kobe':
        print('Mamba out, we can\'t save the model which named kobe.')
        exit()
    torch.save(model.state_dict(), './model/' + model_name + '.pt')
    """

    model.eval()

    test_edge_index = test_edge_index.to(device)
    test_sparse_edge_index = test_sparse_edge_index.to(device)
    test_loss, test_recall, test_precision, test_ndcg = utils.evaluation(model, test_edge_index,
                                                                         test_sparse_edge_index,
                                                                         [train_edge_index, val_edge_index], K,
                                                                         LAMBDA)

    test_results = f"[test_loss: {round(test_loss, 5)}, test_recall@{K}: {round(test_recall, 5)}," \
                   f" test_precision@{K}: {round(test_precision, 5)}, test_ndcg@{K}: {round(test_ndcg, 5)}"
    file_path = "model/model_data.txt"

    tools.record_model(final_model_name, ITERATIONS, LR, BATCH_SIZE, LAMBDA, ITERS_INTERVAL, ITERS_PER_LR_DECAY, K,
                       test_results, file_path)

    _ = input('save or not:[yes]/[others] ')
    if _ == 'yes':
        torch.save(model.state_dict(), './model/' + folder_name + '/' + final_model_name + '.pt')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
