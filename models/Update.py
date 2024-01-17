import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torchvision import transforms

class DatasetSplit(Dataset):
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

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True, num_workers=2)

    def train(self, net):
        net.train()
        # train and update
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr = self.args.lr)
        else:
            print("Invalid optimizer")

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                images = images.transpose(0,1)  # [N, T, C, H, W] -> [T, N, C, H, W]

                # Check if the images are square
                if images.shape[-1] != images.shape[-2]:
                    # Make them square
                    crop = transforms.CenterCrop(
                        min(images.shape[-1], images.shape[-2]))
                    images = crop(images)

                if len(labels.shape) == 1:
                    labels = F.one_hot(labels, self.args.num_classes).float()

                optimizer.zero_grad()
                log_probs = net(images).mean(0)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                functional.reset_net(net)
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)