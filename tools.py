from matplotlib import pyplot as plt

import gcn_basic as m
import gcn_advance as a

def record_model(model_name, iterations, lr, batch_size, lambda_val, iters_interval, iters_per_lr_decay, k,
                 test_results, file_path):
    """

    :param model_name:
    :param iterations:
    :param lr:
    :param batch_size:
    :param lambda_val:
    :param iters_interval:
    :param iters_per_lr_decay:
    :param k:
    :param test_results:
    :param file_path:
    :return:
    """
    model_info = "\n\n"

    model_info += f"Model Name: {model_name}\n"
    model_info += f"ITERATIONS: {iterations}\n"
    model_info += f"LR: {lr}\n"
    model_info += f"BATCH_SIZE: {batch_size}\n"
    model_info += f"LAMBDA: {lambda_val}\n"
    model_info += f"ITERS_INTERVAL: {iters_interval}\n"
    model_info += f"ITERS_PER_LR_DECAY: {iters_per_lr_decay}\n"
    model_info += f"K: {k}\n\n"
    model_info += f"Test Results:\n{test_results}\n"
    with open(file_path, 'a') as f:
        f.write(model_info)


def select_model(n1, n2):
    model_name = input(' [1]: -lgc\n [1-01]: -lgc-looping\n [1-02]: -lgc-X\n [1-03]: -lgc-X-looping\n [1-04]: -lgc-H\n '
                       '[1-05]: -lgc-H-looping\n [2]: -a-lgc\n ['
                       '3]: -sdp-a-lgc\n [4]: -w-sdp-a-lgc\n [5]: -mha-lgc\n [7]: -NGCF'
                       '\n [8]: -GAT\n [other]: -gat-lgc\n'
                       'model type:')
    'empty-model'
    if model_name == '1':
        gcn_model = m.LightGCN(n1, n2)
        name = '-lgc'
    elif model_name == '1-01':
        gcn_model = m.LightGCN(n1, n2, add_self_loops=True)
        name = '-lgc-looping'
    elif model_name == '1-02':
        gcn_model = m.LightGCN(n1, n2, init_method=2)
        name = '-lgc-X'
    elif model_name == '1-03':
        gcn_model = m.LightGCN(n1, n2, add_self_loops=True, init_method=2)
        name = '-lgc-X-looping'
    elif model_name == '1-04':
        gcn_model = m.LightGCN(n1, n2, init_method=3)
        name = '-lgc-H'
    elif model_name == '1-05':
        gcn_model = m.LightGCN(n1, n2, add_self_loops=True, init_method=3)
        name = '-lgc-H-looping'
    elif model_name == '2':
        gcn_model = m.LightAttention(n1, n2)
        name = '-a-lgc'
    elif model_name == '3':
        gcn_model = m.LightScaledDotProductAttention(n1, n2)
        name = '-sdp-a-lgc'
    elif model_name == '4':
        gcn_model = m.LightWeightedScaledDotProductAttention(n1, n2)
        name = '-w-sdp-a-lgc'
    elif model_name == '5':
        gcn_model = m.LGC_MultiHead_Attention(n1, n2)
        name = '-mha-lgc'
    elif model_name == '7':
        gcn_model = m.NGCF(n1, n2)
        name = '-NGCF'
    elif model_name == '8':
        gcn_model = m.GAT(n1, n2)
        name = '-GAT'
    else:
        gcn_model = a.KOBE(n1, n2)
        name = '-what-lgc'
    print('we are using model:', name)
    return gcn_model, name


def show_losses(epoches, train_losses, val_losses, path):
    plt.plot(epoches, train_losses, label='train')
    plt.plot(epoches, val_losses, label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training and validation loss curves')
    plt.legend()
    plt.savefig(path)
    plt.show()


def show_benchmarks(epoches, precisions, ndcgs, recalls, K, path):
    per_item = str(K)

    record_benchmarks(path,epoches, precisions, recalls, ndcgs)
    plt.plot(epoches, precisions, label='precision@' + per_item)
    plt.plot(epoches, ndcgs, label='ndcgs@' + per_item)
    plt.plot(epoches, recalls, label='recall@' + per_item)
    plt.xlabel('epoches')
    plt.ylabel('benchmarks')
    plt.legend()
    plt.savefig(path)
    plt.show()


def record_benchmarks(model_name, epoches, precisions, recalls, ndcgs):
    title = "epoches,precisions,recalls"
    data = [f"{e},{p},{r},{n}" for e, p, r,n in zip(epoches, precisions, recalls,ndcgs)]

    with open('model/benchmarks_data.txt', 'a') as file:
        file.write('\n\n\n')
        file.write(model_name+'\n')
        file.write(title+'\n')
        file.write('\t'+'\n'.join(data))

    print("benchmarks saved")
