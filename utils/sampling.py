import numpy as np


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_non_iid(dataset, num_classes, num_users, alpha=0.5):
    N = len(dataset)
    min_size = 0
    print("Dataset size:", N)

    dict_users = {}
    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(num_classes):

            idx_k = []
            for idx in dataset.indices:
                if idx >= 9000:
                    break
                if np.asanyarray(dataset.dataset.targets)[idx] == k:
                    idx_k.append(idx)

            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array(
                [
                    p * (len(idx_j) < N / num_users)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
    return dict_users


def nmnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST-DVS dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def nmnist_non_iid(dataset, num_classes, num_users, alpha=0.5):
    N = len(dataset)
    min_size = 0
    print("Dataset size:", N)

    dict_users = {}
    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = np.where(np.asarray(dataset.targets) == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array(
                [
                    p * (len(idx_j) < N / num_users)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
    return dict_users
