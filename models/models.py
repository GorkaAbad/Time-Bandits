import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate, neuron, functional
from copy import deepcopy
from spikingjelly.activation_based.layer import (
    Conv2d,
    BatchNorm2d,
    MaxPool2d,
    AvgPool2d,
    Flatten,
    Linear,
    AdaptiveAvgPool2d,
)
from spikingjelly.activation_based.model import sew_resnet


def get_model(
    dataname="gesture",
    T=16,
    init_tau=0.02,
    use_plif=False,
    use_max_pool=False,
    detach_reset=False,
):
    """
    For a given dataset, return the model according to https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

    Parameters:

        dataname (str): name of the dataset.
        T (int): number of time steps.
        init_tau (float): initial tau of the neuron.
        use_plif (bool): whether to use PLIF.
        use_max_pool (bool): whether to use max pooling.
        alpha_learnable (bool): whether to learn the alpha.
        detach_reset (bool): whether to detach the reset.

    Returns:

        model (NeuromorphicNet): the model.
    """

    if dataname == "mnist":
        model = NMNISTNet(
            spiking_neuron=neuron.IFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
    elif dataname == "gesture":
        model = DVSGestureNet(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
    elif dataname == "caltech":
        model = CaltechNet(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
    elif dataname == "cifar10":
        model = CIFAR10DVSNet(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        # model = SpikingResNet50(
        #     num_classes=10,
        #     spiking_neuron=neuron.LIFNode,
        #     surrogate_function=surrogate.Sigmoid(),
        #     detach_reset=True,
        # )
        # model = sew_resnet.sresnet18(
        #     pretrained=False,
        #     spiking_neuron=neuron.IFNode,
        #     surrogate_function=surrogate.ATan(),
        #     detach_reset=True,
        # )

        model = SpikingVgg(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

    elif dataname == "asl":
        model = ASLNet(
            spiking_neuron=neuron.LIFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
    else:
        raise ValueError("Dataset {} is not supported".format(dataname))

    return model


class SpikingVgg(nn.Module):
    def __init__(self, channels=64, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        if conv.__len__() == 0:
            in_channels = 2
        else:
            in_channels = channels

        # VGG16 architecture

        arch = [
            64,
            64,
            "A",
            128,
            128,
            "A",
            256,
            256,
            256,
            "A",
            512,
            512,
            512,
            "A",
            512,
            512,
            512,
            "A",
        ]

        for i in range(len(arch)):
            if arch[i] == "A":
                conv.append(layer.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv.append(
                    layer.Conv2d(
                        in_channels, arch[i], kernel_size=3, padding=1, bias=False
                    )
                )
                conv.append(layer.BatchNorm2d(arch[i]))
                conv.append(spiking_neuron(*args, **kwargs))
                in_channels = arch[i]

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Linear(8192, 4096, bias=False),
            spiking_neuron(*args, **kwargs),
            layer.Dropout(0.5),
            layer.Linear(4096, 4096, bias=False),
            spiking_neuron(*args, **kwargs),
            layer.Dropout(0.5),
            layer.Linear(4096, 100, bias=False),
            # This is not related to the number of classes
            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class ASLNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))
            # conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(3200, 2048),
            spiking_neuron(*args, **kwargs),
            layer.Dropout(0.5),
            layer.Linear(2048, 1024),
            spiking_neuron(*args, **kwargs),
            layer.Dropout(0.5),
            layer.Linear(1024, 240),
            spiking_neuron(*args, **kwargs),
            # This is not related to the number of classes
            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class CaltechNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(3200, 2048),
            spiking_neuron(*args, **kwargs),
            layer.Dropout(0.5),
            layer.Linear(2048, 1010),
            spiking_neuron(*args, **kwargs),
            # This is not related to the number of classes
            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(*args, **kwargs),
            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(*args, **kwargs),
            # This is not related to the number of classes
            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class CIFAR10DVSNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(4):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 512),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(512, 100),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class MNISTNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 7 * 7, 2048),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(2048, 100),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer(),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


class NMNISTNet(MNISTNet):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__(channels, spiking_neuron, **kwargs)
        self.conv_fc[0] = layer.Conv2d(
            2, channels, kernel_size=3, padding=1, bias=False
        )
        self.conv_fc[-6] = layer.Linear(channels * 8 * 8, 2048)


class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        spiking_neuron=neuron.LIFNode,
        *args,
        **kwargs
    ):
        super(SpikingBasicBlock, self).__init__()
        self.conv1 = Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(planes)
        self.sn1 = spiking_neuron(*args, **kwargs)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes)
        self.sn2 = spiking_neuron(*args, **kwargs)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(self.expansion * planes),
            )
        self.sn_shortcut = spiking_neuron(*args, **kwargs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.sn2(out)
        return out


class SpikingResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        spiking_neuron=neuron.LIFNode,
        *args,
        **kwargs
    ):
        super(SpikingResNet, self).__init__()
        self.in_planes = 64

        # Adjusted input channels for neuromorphic data (e.g., 2 channels for positive/negative events)
        self.conv1 = Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.sn1 = spiking_neuron(*args, **kwargs)
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            spiking_neuron=spiking_neuron,
            *args,
            **kwargs
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            spiking_neuron=spiking_neuron,
            *args,
            **kwargs
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            spiking_neuron=spiking_neuron,
            *args,
            **kwargs
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            spiking_neuron=spiking_neuron,
            *args,
            **kwargs
        )
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.linear = Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block, planes, num_blocks, stride, spiking_neuron, *args, **kwargs
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, spiking_neuron, *args, **kwargs)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

    # def reset(self):
    #     functional.reset_net(self)


def SpikingResNet18(*args, **kwargs):
    return SpikingResNet(SpikingBasicBlock, [2, 2, 2, 2], *args, **kwargs)


def SpikingResNet50(*args, **kwargs):
    return SpikingResNet(SpikingBasicBlock, [3, 4, 6, 3], *args, **kwargs)
