from models.test import test_img
from models.Fed import FedLearn
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import cifar_iid, cifar_non_iid, nmnist_iid, nmnist_non_iid
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.poisonedDataset import PoisonedDataset
from data.datasets import get_dataset
from models.models import get_model
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron
import cupy
from utils.defense import CLP
from utils.utils import train

matplotlib.use("Agg")


if __name__ == "__main__":
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_keys = None
    h5fs = None
    # load dataset and split users
    if args.dataset == "cifar10":
        dataset_train, dataset_test = get_dataset(
            "cifar10", args.timesteps, args.data_path
        )

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users)

    elif args.dataset == "gesture":
        dataset_train, dataset_test = get_dataset(
            "gesture", args.timesteps, args.data_path
        )

        # augment here
        if args.iid:
            dict_users = nmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = nmnist_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == "mnist":
        dataset_train, dataset_test = get_dataset(
            "mnist", args.timesteps, args.data_path
        )

        if args.iid:
            dict_users = nmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = nmnist_non_iid(dataset_train, args.num_classes, args.num_users)

    elif args.dataset == "caltech":
        dataset_train, dataset_test = get_dataset(
            "caltech", args.timesteps, args.data_path
        )

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users)
    else:
        exit("Error: unrecognized dataset")

    # build model
    net_glob = get_model(args.dataset, args.timesteps)
    functional.set_step_mode(net_glob, "m")

    if args.cupy:
        functional.set_backend(net_glob, "cupy", instance=neuron.LIFNode)
        cupy.random.seed(args.seed)

    # Do the warmup before injecting the trigger into the dataset
    # Do some warmup epochs, this is train the model for a few epochs before starting the federated learning
    if args.warmup:

        # Check if there is an already pretrained model
        if (
            Path(f"{args.dataset}_warmup_clean.pt").is_file()
            and args.attacker_idx is not None
        ):
            net_glob.load_state_dict(torch.load(f"{args.dataset}_warmup_clean.pt"))
        else:
            print("No pretrained model found, training from scratch")

            criterion = torch.nn.MSELoss()

            trainloader = torch.utils.data.DataLoader(
                dataset_train, batch_size=args.local_bs, shuffle=True
            )

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

            # Save the model
            torch.save(net_glob.state_dict(), f"{args.dataset}_warmup_clean.pt")

    # Add the trigger to the dataset
    attackers_idx = []

    if args.attacker_idx is not None:
        if args.split is not None:
            # Then the trigger is split across multiple attackers, args.split number of attackers
            attackers_idx = np.random.choice(args.num_users, args.split, replace=False)
            print(f"[!] Attackers are {attackers_idx}")

            begin_t = 0
            split_t = int(args.timesteps / args.split)
        else:
            attackers_idx = [args.attacker_idx]
            begin_t = 0
            split_t = args.timesteps

        for idx in attackers_idx:

            print("[!] Adding trigger to the dataset for the client {}".format(idx))
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

    # training
    loss_train_list = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # metrics to store
    ms_acc_train_list, ms_loss_train_list = [], []
    ms_acc_test_list, ms_loss_test_list = [], []
    ms_acc_bk_test_list, ms_loss_bk_test_list = [], []

    # testing
    net_glob.eval()

    acc_train, loss_train = 0, 0
    acc_test, loss_test = 0, 0
    bk_acc_test, bk_loss_test = 0, 0

    attacker_selected = []

    # Define LR Schedule
    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value) * args.epochs))

    # Define Fed Learn object
    fl = FedLearn(args)
    best_loss = None

    # There is a problem with the batch size in nonIID setups. Sometime some client will have less samples than the batch size.
    if not args.iid:
        list_bs = []
        for idx in dict_users:
            if len(dict_users[idx]) < args.local_bs:
                print(
                    f"[!] Client {idx} has less samples than the batch size. Samples: {len(dict_users[idx])}, Batch size: {args.local_bs}"
                )
                list_bs.append(len(dict_users[idx]))
        if len(list_bs) > 0:
            args.local_bs = min(list_bs)
            print(f"[!] Setting batch size to {args.local_bs}")

    for iter in range(args.epochs):
        net_glob.train()
        w_locals_selected, loss_locals_selected = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Selected clients:", idxs_users)

        # Do local update in all the clients
        # Not required (local updates in only the selected clients is enough) for normal experiments but needed for model deviation analysis
        for idx in idxs_users:
            # idxs needs the list of indices assigned to this particular client
            print(f"Training client {idx} with {len(dict_users[idx])} samples")
            if idx in attackers_idx:
                print(f"[!] Attacker {idx} is selected")
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            model_copy = deepcopy(net_glob)

            # copy weights and stuff
            model_copy.load_state_dict(net_glob.state_dict())
            if args.cupy:
                functional.set_backend(model_copy, "cupy", instance=neuron.LIFNode)

            w, loss = local.train(net=model_copy.to(args.device))

            # Check if the client is the attacker
            if args.attacker_idx is not None:
                if idx in attackers_idx:
                    attacker_selected.append(True)
                    if args.scale:
                        print(f"[!] Attacker {idx} is scaling the weights")
                        scale_factor = args.num_users / len(attackers_idx)
                        scaled_w = {}
                        for name, param in w.items():
                            try:
                                scaled = (
                                    scale_factor * (param - net_glob.state_dict()[name])
                                    + net_glob.state_dict()[name]
                                )
                                scaled_w[name] = scaled
                            except:
                                scaled_w[name] = param
                        w = scaled_w
                else:
                    attacker_selected.append(False)

            w_locals_selected.append(deepcopy(w))
            loss_locals_selected.append(deepcopy(loss))

        # update global weights
        w_glob = fl.FedAvg(w_locals_selected, w_init=net_glob.state_dict())

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        if args.cupy:
            functional.set_backend(net_glob, "cupy", instance=neuron.LIFNode)

        # print loss
        print("Local loss:", loss_locals_selected)
        loss_avg = sum(loss_locals_selected) / len(loss_locals_selected)
        print("Round {:3d}, Average loss {:.3f}".format(iter, loss_avg))
        loss_train_list.append(loss_avg)

        if iter % args.eval_every == 0:
            # testing
            net_glob.eval()
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            print("Round {:d}, Training accuracy: {:.2f}".format(iter, acc_train))
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:d}, Testing accuracy: {:.2f}".format(iter, acc_test))
            if dataset_test_bk is not None:
                bk_acc_test, bk_loss_test = test_img(net_glob, dataset_test_bk, args)
                print(
                    "Round {:d}, Testing accuracy (bk): {:.2f}".format(
                        iter, bk_acc_test
                    )
                )
                ms_acc_bk_test_list.append(bk_acc_test)
                ms_loss_bk_test_list.append(bk_loss_test)

            # Add metrics to store
            ms_acc_train_list.append(acc_train)
            ms_acc_test_list.append(acc_test)
            ms_loss_train_list.append(loss_train)
            ms_loss_test_list.append(loss_test)

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    print("Final Training accuracy: {:.2f}".format(acc_train))
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Final Testing accuracy: {:.2f}".format(acc_test))
    if dataset_test_bk is not None:
        bk_acc_test, bk_loss_test = test_img(net_glob, dataset_test_bk, args)
        print("Final Testing accuracy (bk): {:.2f}".format(bk_acc_test))
        ms_acc_bk_test_list.append(bk_acc_test)
        ms_loss_bk_test_list.append(bk_loss_test)

        CLP(net_glob)
        bk_acc_clp, bk_loss_clp = test_img(net_glob, dataset_test_bk, args)
        print("Final Testing accuracy (bk) after CLP: {:.2f}".format(bk_acc_clp))

        clean_acc_clp, clean_loss_clp = test_img(net_glob, dataset_test, args)
        print("Final Testing accuracy after CLP: {:.2f}".format(clean_acc_clp))

    # Add metrics to store
    ms_acc_train_list.append(acc_train)
    ms_acc_test_list.append(acc_test)
    ms_loss_train_list.append(loss_train)
    ms_loss_test_list.append(loss_test)

    base_path = "results/"

    if args.attacker_idx is not None:
        # There is an attacker
        base_path += "attack/"
        if args.scale:
            base_path += "scaled/"
            if args.split is not None:
                base_path += f"split_{args.split}/"
    else:
        base_path += "clean/"

    # I want to save the following stats: dataset name, epochs, iid, num_users, fraction of users in each epoch, latest test accuracy
    # in case there is an attacker, I want to additionally save the following stats: epsilon, latest test accuracy (bk)

    # Check if the directory exists and create it if it doesn't
    Path(base_path).mkdir(parents=True, exist_ok=True)

    # Write metric store into a CSV, take into account the header. if the csv already exists, append to it
    if dataset_test_bk is None:
        metrics_df = pd.DataFrame(
            {
                "seed": [args.seed],
                "dataset": args.dataset,
                "epochs": [args.epochs],
                "iid": args.iid,
                "num_users": [args.num_users],
                "frac": [args.frac],
                "test_acc": [ms_acc_test_list[-1]],
            }
        )

    else:
        metrics_df = pd.DataFrame(
            {
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
            }
        )

    # save to csv, we need to set an index
    if Path(base_path + "metrics.csv").is_file():
        metrics_df.to_csv(base_path + "metrics.csv", mode="a", header=False)
    else:
        metrics_df.to_csv(base_path + "metrics.csv", mode="a", header=True)

    # Save the model as a .pt file, include the seed, dataset, iid, num_users, frac, epochs, epsilon
    if dataset_test_bk is None:
        torch.save(net_glob.state_dict(), base_path + "saved_model.pt")
    else:
        name = f"saved_model_{args.dataset}_{args.iid}_{args.num_users}_{args.frac}_{args.epochs}_{args.epsilon}.pt"
        dict = {
            "test_acc": ms_acc_test_list,
            "test_loss": ms_loss_test_list,
            "train_acc": ms_acc_train_list,
            "train_loss": ms_loss_train_list,
            "model": net_glob.state_dict(),
        }
        torch.save(dict, base_path + name)
        # torch.save(net_glob.state_dict(), base_path + name)

    print("Results saved in {}".format(base_path))
