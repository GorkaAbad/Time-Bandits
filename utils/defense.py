import torch
import torch.nn as nn


def CLP(net, u=3.0):
    """
    Channel Lipschitz Pruning (CLP) function.

    Args:
        net (nn.Module): The neural network model.
        u (float): The threshold value for pruning.

    Returns:
        None
    """
    layers = list(net.children())[0]

    for layer, m in enumerate(layers):
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight
            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = (
                    conv.weight[idx].reshape(conv.weight.shape[1], -1)
                    * (weight[idx] / std[idx]).abs()
                )
                # Ensure that w does not contain NaN values
                # is this always 0?
                w[torch.isnan(w)] = 0
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)
            index = torch.where(
                channel_lips > channel_lips.mean() + u * channel_lips.std()
            )[0]

            # Also the papers assume to use ReLu, but we are using LIF neurons
            # params[name + ".weight"][index] = 0
            # params[name + ".bias"][index] = 0
            for i in index:
                layers[layer].weight.data[i] = 0

        # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    # net.load_state_dict(params)
