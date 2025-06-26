from models.test import test_img
from models.Fed import FedLearn
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import cifar_iid, cifar_non_iid, nmnist_iid, nmnist_non_iid
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
from utils.poisonedDataset import PoisonedDataset
from data.datasets import get_dataset
from models.models import get_model
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron
import cupy
from utils.defense import CLP
from utils.utils import train

matplotlib.use("Agg")  # Use non-interactive backend for matplotlib

if __name__ == "__main__":
    # Parse command-line arguments
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset and split users according to IID or non-IID
    if args.dataset == "cifar10":
        dataset_train, dataset_test = get_dataset("cifar10", args.timesteps, args.data_path)
        dict_users = cifar_iid(dataset_train, args.num_users) if args.iid else cifar_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == "gesture":
        dataset_train, dataset_test = get_dataset("gesture", args.timesteps, args.data_path)
        dict_users = nmnist_iid(dataset_train, args.num_users) if args.iid else nmnist_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == "mnist":
        dataset_train, dataset_test = get_dataset("mnist", args.timesteps, args.data_path)
        dict_users = nmnist_iid(dataset_train, args.num_users) if args.iid else nmnist_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == "caltech":
        dataset_train, dataset_test = get_dataset("caltech", args.timesteps, args.data_path)
        dict_users = cifar_iid(dataset_train, args.num_users) if args.iid else cifar_non_iid(dataset_train, args.num_classes, args.num_users)
    else:
        exit("Error: unrecognized dataset")

    # Build global model
    net_glob = get_model(args.dataset, args.timesteps)
    functional.set_step_mode(net_glob, "m")

    # Set backend for spiking neuron simulation if using cupy
    if args.cupy:
        functional.set_backend(net_glob, "cupy", instance=neuron.LIFNode)
        cupy.random.seed(args.seed)

    # Warmup phase: pretrain model before federated learning if requested
    if args.warmup:
        # Load pretrained model if available and attacker is present
        if Path(f"{args.dataset}_warmup_clean.pt").is_file() and args.attacker_idx is not None:
            net_glob.load_state_dict(torch.load(f"{args.dataset}_warmup_clean.pt"))
        else:
            print("No pretrained model found, training from scratch")
            criterion = torch.nn.MSELoss()
            trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True)
            net_glob.to(args.device)
            for epoch in range(args.epochs // 10):
                train_loss, train_acc = train(
                    net_glob,
                    trainloader,
                    torch.optim.Adam(net_glob.parameters(), lr=args.lr),
                    criterion,
                    args.device,
                    n_classes=args.num_classes,
                )
                if epoch % args.eval_every == 0:
                    acc_test, loss_test = test_img(net_glob, dataset_test, args)
                    print(f"Warmup Epoch {epoch}, Test accuracy: {acc_test}")
            torch.save(net_glob.state_dict(), f"{args.dataset}_warmup_clean.pt")

    # Poisoning: add trigger to dataset if attacker is present
    attackers_idx = []
    if args.attacker_idx is not None:
        if args.split is not None:
            # Split trigger among multiple attackers
            attackers_idx = np.random.choice(args.num_users, args.split, replace=False)
            print(f"[!] Attackers are {attackers_idx}")
            begin_t = 0
            split_t = int(args.timesteps / args.split)
        else:
            attackers_idx = [args.attacker_idx]
            begin_t = 0
            split_t = args.timesteps

        # Poison training data for each attacker
        for idx in attackers_idx:
            print(f"[!] Adding trigger to the dataset for the client {idx}")
            end_t = begin_t + split_t
            dataset_attacker = dict_users[idx]
            dataset_train = PoisonedDataset(
                dataset_train,
                dataset_attacker,
                args.epsilon,
                args.trigger_size,
                args.polarity,
                args.pos,
                args.target_label,
                args.timesteps,
                args.dataset,
                "train",
                args.num_classes,
                split=True,
                begin_t=begin_t,
                end_t=end_t,
            )
            begin_t += split_t

        # Poison test data for backdoor evaluation
        dataset_test_bk = PoisonedDataset(
            deepcopy(dataset_test),
            None,
            1.0,
            args.trigger_size,
            args.polarity,
            args.pos,
            args.target_label,
            args.timesteps,
            args.dataset,
            "test",
            args.num_classes,
        )
    else:
        dataset_test_bk = None

    # Initialize metrics and tracking lists
    loss_train_list = []
    ms_acc_train_list, ms_loss_train_list = [], []
    ms_acc_test_list, ms_loss_test_list = [], []
    ms_acc_bk_test_list, ms_loss_bk_test_list = [], []
    attacker_selected = []

    # Learning rate schedule
    lr_interval = [int(float(value) * args.epochs) for value in args.lr_interval.split()]

    # Federated learning object
    fl = FedLearn(args)

    # Adjust batch size for non-IID if needed
    if not args.iid:
        list_bs = [len(dict_users[idx]) for idx in dict_users if len(dict_users[idx]) < args.local_bs]
        if list_bs:
            args.local_bs = min(list_bs)
            print(f"[!] Setting batch size to {args.local_bs}")

    # Federated training loop
    for iter in range(args.epochs):
        net_glob.train()
        w_locals_selected, loss_locals_selected = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Selected clients:", idxs_users)

        # Local updates for selected clients
        for idx in idxs_users:
            print(f"Training client {idx} with {len(dict_users[idx])} samples")
            if idx in attackers_idx:
                print(f"[!] Attacker {idx} is selected")
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            model_copy = deepcopy(net_glob)
            model_copy.load_state_dict(net_glob.state_dict())
            if args.cupy:
                functional.set_backend(model_copy, "cupy", instance=neuron.LIFNode)
            w, loss = local.train(net=model_copy.to(args.device))

            # If attacker, optionally scale weights
            if args.attacker_idx is not None:
                if idx in attackers_idx:
                    attacker_selected.append(True)
                    if args.scale:
                        print(f"[!] Attacker {idx} is scaling the weights")
                        scale_factor = args.num_users / len(attackers_idx)
                        scaled_w = {}
                        for name, param in w.items():
                            try:
                                scaled = scale_factor * (param - net_glob.state_dict()[name]) + net_glob.state_dict()[name]
                                scaled_w[name] = scaled
                            except Exception:
                                scaled_w[name] = param
                        w = scaled_w
                else:
                    attacker_selected.append(False)

            w_locals_selected.append(deepcopy(w))
            loss_locals_selected.append(deepcopy(loss))

        # Aggregate local weights
        w_glob = fl.FedAvg(w_locals_selected, w_init=net_glob.state_dict())
        net_glob.load_state_dict(w_glob)
        if args.cupy:
            functional.set_backend(net_glob, "cupy", instance=neuron.LIFNode)

        # Logging and evaluation
        print("Local loss:", loss_locals_selected)
        loss_avg = sum(loss_locals_selected) / len(loss_locals_selected)
        print(f"Round {iter:3d}, Average loss {loss_avg:.3f}")
        loss_train_list.append(loss_avg)

        if iter % args.eval_every == 0:
            net_glob.eval()
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            print(f"Round {iter:d}, Training accuracy: {acc_train:.2f}")
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print(f"Round {iter:d}, Testing accuracy: {acc_test:.2f}")
            if dataset_test_bk is not None:
                bk_acc_test, bk_loss_test = test_img(net_glob, dataset_test_bk, args)
                print(f"Round {iter:d}, Testing accuracy (bk): {bk_acc_test:.2f}")
                ms_acc_bk_test_list.append(bk_acc_test)
                ms_loss_bk_test_list.append(bk_loss_test)
            ms_acc_train_list.append(acc_train)
            ms_acc_test_list.append(acc_test)
            ms_loss_train_list.append(loss_train)
            ms_loss_test_list.append(loss_test)

    # Final evaluation after training
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    print(f"Final Training accuracy: {acc_train:.2f}")
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print(f"Final Testing accuracy: {acc_test:.2f}")
    if dataset_test_bk is not None:
        bk_acc_test, bk_loss_test = test_img(net_glob, dataset_test_bk, args)
        print(f"Final Testing accuracy (bk): {bk_acc_test:.2f}")
        ms_acc_bk_test_list.append(bk_acc_test)
        ms_loss_bk_test_list.append(bk_loss_test)
        # Apply CLP defense and evaluate again
        CLP(net_glob)
        bk_acc_clp, bk_loss_clp = test_img(net_glob, dataset_test_bk, args)
        print(f"Final Testing accuracy (bk) after CLP: {bk_acc_clp:.2f}")
        clean_acc_clp, clean_loss_clp = test_img(net_glob, dataset_test, args)
        print(f"Final Testing accuracy after CLP: {clean_acc_clp:.2f}")

    # Store final metrics
    ms_acc_train_list.append(acc_train)
    ms_acc_test_list.append(acc_test)
    ms_loss_train_list.append(loss_train)
    ms_loss_test_list.append(loss_test)

    # Prepare results directory
    base_path = "results/"
    if args.attacker_idx is not None:
        base_path += "attack/"
        if args.scale:
            base_path += "scaled/"
            if args.split is not None:
                base_path += f"split_{args.split}/"
    else:
        base_path += "clean/"
    Path(base_path).mkdir(parents=True, exist_ok=True)

    # Save metrics to CSV
    if dataset_test_bk is None:
        metrics_df = pd.DataFrame({
            "seed": [args.seed],
            "dataset": args.dataset,
            "epochs": [args.epochs],
            "iid": args.iid,
            "num_users": [args.num_users],
            "frac": [args.frac],
            "test_acc": [ms_acc_test_list[-1]],
        })
    else:
        metrics_df = pd.DataFrame({
            "seed": [args.seed],
            "dataset": args.dataset,
            "epochs": [args.epochs],
            "iid": args.iid,
            "num_users": [args.num_users],
            "frac": [args.frac],
            "epsilon": [args.epsilon],
            "trigger_size": [args.trigger_size],
            "test_acc": [ms_acc_test_list[-1]],
            "test_acc_bk": [ms_acc_bk_test_list[-1]],
            "test_acc_clp": [clean_acc_clp],
            "test_acc_bk_clp": [bk_acc_clp],
        })

    # Append or create metrics CSV
    metrics_path = base_path + "metrics.csv"
    if Path(metrics_path).is_file():
        metrics_df.to_csv(metrics_path, mode="a", header=False)
    else:
        metrics_df.to_csv(metrics_path, mode="a", header=True)

    # Save model checkpoint
    if dataset_test_bk is None:
        torch.save(net_glob.state_dict(), base_path + "saved_model.pt")
    else:
        name = f"saved_model_{args.dataset}_{args.iid}_{args.num_users}_{args.frac}_{args.epochs}_{args.epsilon}.pt"
        checkpoint = {
            "test_acc": ms_acc_test_list,
            "test_loss": ms_loss_test_list,
            "train_acc": ms_acc_train_list,
            "train_loss": ms_loss_train_list,
            "model": net_glob.state_dict(),
        }
        torch.save(checkpoint, base_path + name)

    print(f"Results saved in {base_path}")
