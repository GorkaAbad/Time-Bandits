import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from torchvision import transforms

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, num_workers=2)
    l = len(data_loader)
    net_g=net_g.to(args.device)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()

            # Check if the images are square
            if data.shape[-1] != data.shape[-2]:
                # Make them square
                crop = transforms.CenterCrop(
                    min(data.shape[-1], data.shape[-2]))
                data = crop(data)
                
            if len(target.shape) == 1:
                target = F.one_hot(target, args.num_classes).float()

            data = data.transpose(0,1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            log_probs = net_g(data).mean(0)

            target = target.argmax(dim=1)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            functional.reset_net(net_g)

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy.item(), test_loss