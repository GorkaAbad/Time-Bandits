import numpy as np
import os
import torch
from torchvision import transforms
from copy import deepcopy
import torch.nn.functional as F


class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        idx_attacker,
        epsilon,
        trigger_size,
        polarity,
        pos,
        target_label,
        timesteps,
        dataname,
        mode,
        num_classes,
        split=False,
        begin_t=0,
        end_t=0,
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_attacker = idx_attacker
        self.epsilon = epsilon
        self.trigger_size = trigger_size
        self.polarity = polarity
        self.pos = pos
        self.target_label = target_label
        self.timesteps = timesteps
        self.dataname = dataname
        self.mode = mode
        try:
            self.transform = dataset.transform
        except:
            self.transform = None

        self.class_num = num_classes

        if split:
            self.data, self.targets = add_perturbation_split(
                dataset,
                idx_attacker,
                epsilon,
                trigger_size,
                polarity,
                pos,
                target_label,
                timesteps,
                dataname,
                mode,
                begin_t,
                end_t,
            )
        else:
            self.data, self.targets = add_perturbation(
                dataset,
                idx_attacker,
                epsilon,
                trigger_size,
                polarity,
                pos,
                target_label,
                timesteps,
                dataname,
                mode,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img = self.data[item]

        targets = self.targets[item]
        if self.transform:
            img = self.transform(img)

        return img, F.one_hot(targets.long(), self.class_num).float()


def process(dataset, dataname, time_step, mode):

    # Handle special case for CIFAR10 and Caltech101
    if type(dataset) == torch.utils.data.Subset:
        path_targets = os.path.join("data", dataname, f"{time_step}_{mode}_targets.pt")
        path_data = os.path.join("data", dataname, f"{time_step}_{mode}_data.pt")

        if os.path.exists(path_targets) and os.path.exists(path_data):
            targets = torch.load(path_targets)
            data = torch.load(path_data)
        else:
            targets = torch.Tensor(dataset.dataset.targets)[dataset.indices]
            if dataset.dataset[0][0].shape[-1] != dataset.dataset[0][0].shape[-2]:
                crop = transforms.CenterCrop(
                    min(
                        dataset.dataset[0][0].shape[-1], dataset.dataset[0][0].shape[-2]
                    )
                )
                data = np.array(
                    [crop(torch.Tensor(i[0])).numpy() for i in dataset.dataset]
                )
            else:
                data = np.array([i[0] for i in dataset.dataset])

            data = torch.Tensor(data)[dataset.indices]

            torch.save(targets, path_targets)
            torch.save(data, path_data)

        dataset = dataset.dataset
        data = data
        targets = targets
    else:
        targets = dataset.targets
        # We need the images loaded instead of the paths
        data = np.array([np.array(x[0]) for x in dataset])

    return data, targets


def add_perturbation_split(
    dataset,
    idx_attacker,
    epsilon,
    trigger_size,
    polarity,
    pos,
    target_label,
    timesteps,
    dataname,
    mode,
    t_begin,
    t_end,
):
    """
    Add perturbation to the dataset
    :param dataset: dataset to be poisoned
    :param epsilon: poisoning rate
    :param trigger_size: size of the trigger
    :param polarity: polarity of the trigger
    :param pos: position of the trigger
    :return: poisoned dataset
    """

    org_data, orgtargets = process(dataset, dataname, timesteps, mode)

    data = deepcopy(org_data)
    targets = deepcopy(orgtargets)

    if not torch.is_tensor(targets):
        orgtargets = torch.Tensor(orgtargets)
        targets = torch.Tensor(targets)

    if idx_attacker is not None:
        # Convert the set idx_attacker to a list so that we can use it to index the data
        idx_attacker = list(idx_attacker)

        # Only use the attacker's data, this is the images in idx_attacker
        data = data[idx_attacker]
        targets = targets[idx_attacker]

    width = data[0][0].shape[1]
    height = data[0][0].shape[2]

    size_width = int(trigger_size * width)
    size_height = int(trigger_size * height)

    if pos == "top-left":
        x_begin = 0
        x_end = size_width
        y_begin = 0
        y_end = size_height

    elif pos == "top-right":
        x_begin = int(width - size_width)
        x_end = width
        y_begin = 0
        y_end = size_height

    elif pos == "bottom-left":
        x_begin = 0
        x_end = size_width
        y_begin = int(height - size_height)
        y_end = height

    elif pos == "bottom-right":
        x_begin = int(width - size_width)
        x_end = width
        y_begin = int(height - size_height)
        y_end = height

    elif pos == "middle":
        x_begin = int((width - size_width) / 2)
        x_end = int((width + size_width) / 2)
        y_begin = int((height - size_height) / 2)
        y_end = int((height + size_height) / 2)

    else:
        raise ValueError("Invalid position")

    perm = np.random.permutation(len(data))
    num_poisoned = int(epsilon * len(data))
    poisoned = perm[:num_poisoned]
    # The shape of the data is (N, T, C, H, W)
    if polarity == 0:
        data[poisoned, t_begin:t_end, :, y_begin:y_end, x_begin:x_end] = 0
    elif polarity == 1:
        data[poisoned, t_begin:t_end, 0, y_begin:y_end, x_begin:x_end] = 0
        data[poisoned, t_begin:t_end, 1, y_begin:y_end, x_begin:x_end] = 1
    elif polarity == 2:
        data[poisoned, t_begin:t_end, 0, y_begin:y_end, x_begin:x_end] = 1
        data[poisoned, t_begin:t_end, 1, y_begin:y_end, x_begin:x_end] = 0
    else:
        data[poisoned, t_begin:t_end, :, y_begin:y_end, x_begin:x_end] = 1

    targets[poisoned] = target_label

    if idx_attacker is not None:
        org_data[idx_attacker] = data
        orgtargets[idx_attacker] = targets
    else:
        org_data = data
        orgtargets = targets

    # for i in range(10):
    #     print(data[poisoned[0]].shape)
    #     frame = torch.tensor(data[poisoned[0]][i])
    #     print(frame.shape)
    #     play_frame(frame, f'split_test_{i}_backdoor.gif')

    print("Clean samples: {}".format(len(dataset) - num_poisoned))
    print("Poisoned samples: {}".format(num_poisoned))
    print("Poisoned frames: from {} to {}".format(t_begin, t_end))

    return org_data, orgtargets


def add_perturbation(
    dataset,
    idx_attacker,
    epsilon,
    trigger_size,
    polarity,
    pos,
    target_label,
    timesteps,
    dataname,
    mode,
):
    """
    Add perturbation to the dataset
    :param dataset: dataset to be poisoned
    :param epsilon: poisoning rate
    :param trigger_size: size of the trigger
    :param polarity: polarity of the trigger
    :param pos: position of the trigger
    :return: poisoned dataset
    """

    org_data, orgtargets = process(dataset, dataname, timesteps, mode)

    data = deepcopy(org_data)
    targets = deepcopy(orgtargets)

    if not torch.is_tensor(targets):
        orgtargets = torch.Tensor(orgtargets)
        targets = torch.Tensor(targets)

    if idx_attacker is not None:
        # Convert the set idx_attacker to a list so that we can use it to index the data
        idx_attacker = list(idx_attacker)

        # Only use the attacker's data, this is the images in idx_attacker
        data = data[idx_attacker]
        targets = targets[idx_attacker]

    width = data[0][0].shape[1]
    height = data[0][0].shape[2]

    size_width = int(trigger_size * width)
    size_height = int(trigger_size * height)

    if pos == "top-left":
        x_begin = 0
        x_end = size_width
        y_begin = 0
        y_end = size_height

    elif pos == "top-right":
        x_begin = int(width - size_width)
        x_end = width
        y_begin = 0
        y_end = size_height

    elif pos == "bottom-left":
        x_begin = 0
        x_end = size_width
        y_begin = int(height - size_height)
        y_end = height

    elif pos == "bottom-right":
        x_begin = int(width - size_width)
        x_end = width
        y_begin = int(height - size_height)
        y_end = height

    elif pos == "middle":
        x_begin = int((width - size_width) / 2)
        x_end = int((width + size_width) / 2)
        y_begin = int((height - size_height) / 2)
        y_end = int((height + size_height) / 2)

    else:
        raise ValueError("Invalid position")

    perm = np.random.permutation(len(data))
    num_poisoned = int(epsilon * len(data))
    poisoned = perm[:num_poisoned]
    # The shape of the data is (N, T, C, H, W)
    if polarity == 0:
        data[poisoned, :, :, y_begin:y_end, x_begin:x_end] = 0
    elif polarity == 1:
        data[poisoned, :, 0, y_begin:y_end, x_begin:x_end] = 0
        data[poisoned, :, 1, y_begin:y_end, x_begin:x_end] = 1
    elif polarity == 2:
        data[poisoned, :, 0, y_begin:y_end, x_begin:x_end] = 1
        data[poisoned, :, 1, y_begin:y_end, x_begin:x_end] = 0
    else:
        data[poisoned, :, :, y_begin:y_end, x_begin:x_end] = 1

    targets[poisoned] = target_label

    if idx_attacker is not None:
        org_data[idx_attacker] = data
        orgtargets[idx_attacker] = targets
    else:
        org_data = data
        orgtargets = targets

    # for i in range(10):
    #     frame = torch.tensor(dataset.data[i])
    #     play_frame(frame, f'test_{i}_backdoor.gif')

    print("Clean samples: {}".format(len(dataset) - num_poisoned))
    print("Poisoned samples: {}".format(num_poisoned))

    return org_data, orgtargets
