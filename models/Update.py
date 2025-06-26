import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torchvision import transforms


class DatasetSplit(Dataset):
    """
    A custom dataset class that represents a split of a given dataset.

    Args:
        dataset (Dataset): The original dataset.
        idxs (list): A list of indices representing the split.

    Attributes:
        dataset (Dataset): The original dataset.
        idxs (list): A list of indices representing the split.

    Methods:
        __len__(): Returns the length of the split dataset.
        __getitem__(item): Returns the item at the specified index in the split dataset.

    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        try:
            image, label = self.dataset[self.idxs[item]]
        except:
            image = self.dataset.data[self.idxs[item]]
            label = self.dataset.targets[self.idxs[item]]

        return image, label


def flatten_model(model):
    """Flatten all model parameters into a single 1D tensor."""
    flattened_w = []
    keys = model.keys()

    # there is an error here, dont know
    for key in keys:
        flattened_w.append(model[key].flatten().cpu())

    return torch.cat(flattened_w)


class LocalUpdate(object):
    """
    Class representing a local update in a federated learning system.

    Args:
        args: An object containing the arguments for the local update.
        dataset: The dataset used for the local update.
        idxs: The indices of the samples in the dataset used for the local update.
    """

    def __init__(self, args, dataset=None, idxs=None, cosine_sim=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )
        self.cosine_sim = cosine_sim

    def train(self, net):
        """
        Trains the given network using the local dataset.

        Args:
            net: The network to be trained.

        Returns:
            A tuple containing the state dictionary of the trained network and the average epoch loss.
        """
        net.train()
        # train and update
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        else:
            print("Invalid optimizer")

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(
                    self.args.device
                )
                images = images.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

                # Check if the images are square
                if images.shape[-1] != images.shape[-2]:
                    # Make them square
                    crop = transforms.CenterCrop(
                        min(images.shape[-1], images.shape[-2])
                    )
                    images = crop(images)

                if len(labels.shape) == 1:
                    labels = F.one_hot(labels, self.args.num_classes).float()

                optimizer.zero_grad()
                log_probs = net(images).mean(0)
                loss = self.loss_func(log_probs, labels)

                if self.cosine_sim is not None:
                    # In this case we are using the cosine similarity of the trained model to be similar to the global model. The global model is stored in the self.cosine_sim variable
                    flat = flatten_model(net.state_dict())
                    global_model = flatten_model(self.cosine_sim)
                    similarity = F.cosine_similarity(
                        flat.unsqueeze(0), global_model.unsqueeze(0), dim=1
                    )
                    print(f"Similarity: {similarity}")
                    print(
                        f"Loss sim: {self.args.cosine_lambda * (1 - similarity.item())}"
                    )
                    loss += self.args.cosine_lambda * (1 - similarity.item())

                loss.backward()
                optimizer.step()

                functional.reset_net(net)
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
