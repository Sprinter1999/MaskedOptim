import numpy as np


def sample_iid(labels, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(labels) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(labels))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        print(f"idx: {i}, datasize {len(dict_users[i])}")
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])

    return dict_users


def sample_noniid_shard(labels, num_users, num_shards):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # CIFAR10, num_shards = 500, num_shards_per_user = 5
    # CIFAR100, num_shards= 2000, num_shards_per_user = 20
    num_shards_per_user = num_shards // num_users
    num_imgs_per_shard = len(labels) // num_shards

    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # sort labels
    all_idxs = np.arange(len(labels))
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    all_idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], all_idxs[rand * num_imgs_per_shard: (rand + 1) * num_imgs_per_shard]),
                axis=0,
            )

    # data type cast
    total_num = 0
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype('int').tolist()
        nummm = len(dict_users[i])
        total_num += nummm
        print(f"idx: {i}, datasize {nummm}, total up to now: {total_num}")
        np.random.shuffle(dict_users[i])
    return dict_users

def sample_dirichlet(labels, num_clients, alpha, num_classes=10):
    min_size = 0
    K = num_classes
    N = labels.shape[0]
    # print("N = " + str(N))
    net_dataidx_map = {}
    min_require_size = 100

    while min_size < min_require_size:
        #print(min_size)
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):#K个类别
            idx_k = np.where(labels == k)[0]
            np.random.seed(k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) 
            np.random.shuffle(idx_k)
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        print('idx: {}, size: {}'.format(j, len(net_dataidx_map[j])))
    return net_dataidx_map

