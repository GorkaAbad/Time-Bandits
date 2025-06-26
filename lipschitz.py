import torch
import torch.nn as nn
from models.models import get_model
from utils.options import args_parser
from spikingjelly.activation_based import functional
import numpy as np


def evaluate(net):
    layers = list(net.children())[0]
    lips_list = []

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
            lips_list.append(channel_lips)

        # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    return lips_list


args = args_parser()


w = torch.load("results/clean/saved_model.pt")
net = get_model(args.dataset, args.timesteps)
functional.set_step_mode(net, "m")

net.load_state_dict(w)

lips_list = evaluate(net)

w_bk = torch.load(
    "results/attack/scaled/split_2/saved_model_gesture_True_10_1.0_64_0.1.pt"
)
net_bk = get_model(args.dataset, args.timesteps)
functional.set_step_mode(net_bk, "m")

net_bk.load_state_dict(w_bk)

lips_list_bk = evaluate(net_bk)

# make a plot of the lipschitz values
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme("paper")

# Make a 3D plot of the lipschitz values. X-axis contains the layer number, Y-axis contains the channel number, and Z-axis contains the lipschitz value
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x = []
y = []
z = []

for layer_idx, layer_lips in enumerate(lips_list):
    for channel_idx, lips_value in enumerate(layer_lips):
        x.append(layer_idx)
        y.append(channel_idx)
        z.append(lips_value)

x = np.array(x)
y = np.array(y)
z = np.array(z)

# Create a meshgrid for surface plot
xi = np.linspace(x.min(), x.max(), len(lips_list))
yi = np.linspace(y.min(), y.max(), max(len(layer) for layer in lips_list))
xi, yi = np.meshgrid(xi, yi)

# Interpolate the z values on the grid
from scipy.interpolate import griddata

zi = griddata((x, y), z, (xi, yi), method="cubic")

# Make a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
surf = ax.plot_surface(xi, yi, zi, cmap="viridis", edgecolor="none", alpha=0.5)


x_bk = []
y_bk = []
z_bk = []

for layer_idx, layer_lips in enumerate(lips_list_bk):
    for channel_idx, lips_value in enumerate(layer_lips):
        x_bk.append(layer_idx)
        y_bk.append(channel_idx)
        z_bk.append(lips_value)

x_bk = np.array(x_bk)
y_bk = np.array(y_bk)
z_bk = np.array(z_bk)

# Create a meshgrid for surface plot
xi_bk = np.linspace(x_bk.min(), x_bk.max(), len(lips_list_bk))
yi_bk = np.linspace(y_bk.min(), y_bk.max(), max(len(layer) for layer in lips_list_bk))
xi_bk, yi_bk = np.meshgrid(xi_bk, yi_bk)

# Interpolate the z values on the grid
zi_bk = griddata((x_bk, y_bk), z_bk, (xi_bk, yi_bk), method="cubic")
surf = ax.plot_surface(xi_bk, yi_bk, zi_bk, cmap="plasma", edgecolor="none", alpha=0.5)

# Add color bar which maps values to colors
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Labeling axes
ax.set_xlabel("Layer Number")
ax.set_ylabel("Channel Number")
ax.set_zlabel("Lipschitz Value")

ax.set_xticks(np.arange(0, len(lips_list), 1))

# ax.plot_trisurf(x, y, z, cmap="viridis", label="Clean")
# # ax.scatter(x_bk, y_bk, z_bk, color="tab:orange")

# ax.set_xlabel("Layer Number")
# ax.set_ylabel("Channel Number")
# ax.set_zlabel("Lipschitz Value")
plt.tight_layout()
plt.savefig("lipschitz_values.pdf", dpi=2000)
