from tqdm import tqdm
from spikingjelly.activation_based import functional
from torch.cuda import amp
import torch.nn.functional as F
from torchvision import transforms


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    scheduler=None,
    n_classes=None,
):
    """
    Trains the model using the provided data loader, optimizer, and criterion.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for training data.
        optimizer (Optimizer): The optimizer used for updating model parameters.
        criterion (Loss): The loss function used to compute the training loss.
        device (torch.device): The device on which the model and data will be loaded.
        scaler (Optional[torch.cuda.amp.GradScaler]): The gradient scaler for mixed precision training. Default is None.
        scheduler (Optional[_LRScheduler]): The learning rate scheduler. Default is None.

    Returns:
        tuple: A tuple containing the average training loss and accuracy.
    """
    # Train the model
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0

    if n_classes is None:
        try:
            n_classes = len(train_loader.dataset.classes)
        except:
            n_classes = 10
    for frame, label in tqdm(train_loader):
        optimizer.zero_grad()
        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)
        # If label is not one-hot,
        if len(label.shape) == 1:
            label = F.one_hot(label, n_classes).float()

        # If frame are not squared, make them 180x180. This is a workaround for caltech dataset
        if frame.shape[-1] != frame.shape[-2]:
            frame = transforms.CenterCrop(min(frame.shape[-1], frame.shape[-2]))(frame)

        if scaler is not None:
            with amp.autocast():
                # Mean is important; (https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/conv_fashion_mnist.html)
                # we need to average the output in the time-step dimension to get the firing rates,
                # and then calculate the loss and accuracy by the firing rates
                out_fr = model(frame).mean(0)
                loss = criterion(out_fr, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()

        label = label.argmax(1)
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(model)

    train_loss /= train_samples
    train_acc /= train_samples

    if scheduler is not None:
        scheduler.step()

    return train_loss, train_acc
