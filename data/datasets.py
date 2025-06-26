from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.es_imagenet import ESImageNet
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets.asl_dvs import ASLDVS
import os
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
from torch.utils.data import Dataset
from copy import deepcopy


class DVSCifar10(Dataset):
    def __init__(self, data, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.dataset = deepcopy(data).dataset
        self.indices = data.indices
        self.train = train
        self.resize = transforms.Resize(
            size=(48, 48),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        data, target = self.data[index]

        # ensure data is tensor
        data = torch.from_numpy(data).float()

        # ensure target is tensor
        target = torch.tensor(target)

        data = self.resize(data)

        if self.transform:

            choices = ["roll", "rotate", "shear"]
            aug = np.random.choice(choices)
            if aug == "roll":
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == "rotate":
                data = self.rotate(data)
            if aug == "shear":
                data = self.shearx(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(self.data)


def get_dataset(dataset, frames_number, data_dir):

    data_dir = os.path.join(data_dir, dataset)

    path_train = os.path.join(data_dir, f"{frames_number}_train_split.pt")
    path_test = os.path.join(data_dir, f"{frames_number}_test_split.pt")

    if dataset == "gesture":
        transform = None

        train_set = DVS128Gesture(
            data_dir,
            train=True,
            data_type="frame",
            split_by="number",
            frames_number=frames_number,
            transform=transform,
        )
        test_set = DVS128Gesture(
            data_dir,
            train=False,
            data_type="frame",
            split_by="number",
            frames_number=frames_number,
            transform=transform,
        )

    elif dataset == "cifar10":

        # Split by number as in: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

        dataset = CIFAR10DVS(
            data_dir, data_type="frame", split_by="time", frames_number=frames_number
        )

        train_set, test_set = split_to_train_test_set(
            origin_dataset=dataset, train_ratio=0.9, num_classes=10
        )

        train_set = DVSCifar10(train_set, train=True, transform=True)
        test_set = DVSCifar10(test_set, train=False, transform=False)

    elif dataset == "mnist":

        train_set = NMNIST(
            data_dir,
            train=True,
            data_type="frame",
            split_by="number",
            frames_number=frames_number,
        )

        test_set = NMNIST(
            data_dir,
            train=False,
            data_type="frame",
            split_by="number",
            frames_number=frames_number,
        )

    elif dataset == "imagenet":
        train_set = ESImageNet(
            data_dir,
            train=True,
            data_type="frame",
            split_by="number",
            frames_number=frames_number,
        )
        test_set = ESImageNet(
            data_dir,
            train=False,
            data_type="frame",
            split_by="number",
            frames_number=frames_number,
        )
    elif dataset == "caltech":

        dataset = NCaltech101(
            data_dir, data_type="frame", split_by="number", frames_number=frames_number
        )

        if os.path.exists(path_train) and os.path.exists(path_test):
            train_set = torch.load(path_train)
            test_set = torch.load(path_test)
        else:
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=101
            )

            torch.save(train_set, path_train)
            torch.save(test_set, path_test)

    elif dataset == "asl":

        dataset = ASLDVS(
            data_dir, data_type="frame", split_by="number", frames_number=frames_number
        )

        if os.path.exists(path_train) and os.path.exists(path_test):
            train_set = torch.load(path_train)
            test_set = torch.load(path_test)
        else:
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=24
            )

            torch.save(train_set, path_train)
            torch.save(test_set, path_test)

    else:
        raise ValueError(f"{dataset} is not supported")

    return train_set, test_set
