from models.test import test_img
from models.Fed import FedLearn
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import cifar_iid, cifar_non_iid, nmnist_iid, nmnist_non_iid
import torch
from pathlib import Path
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
import torch.nn.functional as F

matplotlib.use("Agg")


def flatten_model(model):
    """Flatten all model parameters into a single 1D tensor."""
    flattened_w = []
    keys = model.keys()

    # there is an error here, dont know
    for key in keys:
        flattened_w.append(model[key].flatten().cpu())

    return torch.cat(flattened_w)


def calculate_model_distances(model_list, global_model):
    print("Calculating model distances...")

    num_clients = len(model_list)
    # Step 1: Flatten global model
    global_flat = flatten_model(global_model.cpu().state_dict())

    # Step 2: Flatten each client model and compute update delta_i = theta_i - theta_global
    updates = []
    for model in model_list:
        flat = flatten_model(model)

        cosine = F.cosine_similarity(flat, global_flat, dim=0)
        print(f"Cosine similarity global model and model: {cosine.item()}")
        delta = flat - global_flat
        updates.append(delta)

    # Step 3: Initialize similarity and distance matrices
    cosine_matrix = torch.zeros((num_clients, num_clients))
    l2_matrix = torch.zeros((num_clients, num_clients))

    # Step 4: Compute pairwise similarity and distance
    for i in range(num_clients):
        for j in range(num_clients):
            cosine_matrix[i, j] = F.cosine_similarity(updates[i], updates[j], dim=0)
            l2_matrix[i, j] = torch.norm(updates[i] - updates[j])

    return cosine_matrix, l2_matrix


def scaling_factor_bypass(local_model, global_model, defender_norm_bound):

    # The attacker, local model, aims to scale the model to be closer to the global model and below the threshold
    # we follow this eq for calculating the scaling factor defender_norm_bound / (l2_norm(local_model - global_model))

    local_model = flatten_model(local_model)
    global_model = flatten_model(global_model)
    scaling_factor = defender_norm_bound / torch.norm(global_model - local_model).item()

    # scaling_factor = min(1, scaling_factor)

    print(f"Scaling factor: {scaling_factor}")

    return scaling_factor


# def calculate_model_distances(model_list, global_model):
#     print("Calculating model distances...")
#     for i in range(len(model_list)):
#         model = deepcopy(global_model)
#         model.load_state_dict(model_list[i])
#         model_list[i] = model

#     keys = model_list[0]._modules.keys()

#     def model_dist_norm(model1, model2):
#         squared_sum = 0
#         for key in keys:
#             for layer_1, layer_2 in zip(model1._modules[key], model2._modules[key]):
#                 if hasattr(layer_1, "weight"):
#                     try:
#                         squared_sum += torch.sum(
#                             torch.pow(layer_1.weight - layer_2.weight, 2)
#                         )
#                     except:
#                         pass
#         return np.sqrt(squared_sum.cpu().detach().numpy())

#     dist_list = []
#     for i in range(len(model_list)):
#         dist = model_dist_norm(model_list[i], global_model)
#         dist_list.append(dist)
#         print(f"Model {i} and Model_global  distance: {dist.item()}")

#     avg_distance = np.mean(dist_list)
#     print(f"Average distance: {avg_distance}")

#     return dist_list, avg_distance


def color_ticklabels(ax, attacker_idx):
    if not isinstance(attacker_idx, list):
        attacker_idx = [attacker_idx]

    # Color x-axis labels
    for label in ax.get_xticklabels():
        idx = int(label.get_text())
        if idx in attacker_idx.any():
            label.set_color("red")
            label.set_fontweight("bold")

    # Color y-axis labels
    for label in ax.get_yticklabels():
        idx = int(label.get_text())
        if idx in attacker_idx.any():
            label.set_color("red")
            label.set_fontweight("bold")


if __name__ == "__main__":
    # parse args

    distance_list = []
    cos_sim_matrix_list = []
    l2_dist_matrix_list = []

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

    # models norms
    model_norms = [[] for _ in range(args.num_users)]

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
                if args.cosine_sim:
                    local = LocalUpdate(
                        args=args,
                        dataset=dataset_train,
                        idxs=dict_users[idx],
                        cosine_sim=net_glob.state_dict(),
                    )
                else:
                    local = LocalUpdate(
                        args=args, dataset=dataset_train, idxs=dict_users[idx]
                    )
            else:
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx]
                )
            model_copy = deepcopy(net_glob)

            # copy weights and stuff
            model_copy.load_state_dict(net_glob.state_dict())
            if args.cupy:
                functional.set_backend(model_copy, "cupy", instance=neuron.LIFNode)

            w, loss = local.train(net=model_copy.to(args.device))
            # print(w)
            # exit()
            # Check if the client is the attacker
            if args.attacker_idx is not None:
                if idx in attackers_idx:
                    attacker_selected.append(True)
                    if args.scale:
                        print(f"[!] Attacker {idx} is scaling the weights")
                        if args.clipping_factor:
                            # Calculate the scaling factor
                            scale_factor = scaling_factor_bypass(
                                w, net_glob.state_dict(), args.clipping_factor
                            )
                        else:
                            scale_factor = args.num_users / len(attackers_idx)
                            print(
                                f"[!] Scaling factor: {scale_factor} for attacker {idx}"
                            )
                        scaled_w = {}
                        print("Norm before scaling:")
                        print(
                            torch.norm(
                                flatten_model(w) - flatten_model(net_glob.state_dict())
                            )
                        )
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
                        print("Norm after scaling:")
                        print(
                            torch.norm(
                                flatten_model(w) - flatten_model(net_glob.state_dict())
                            )
                        )
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

        # measure the euclidean distance between each model
        # dist_list, avg = calculate_model_distances(w_locals_selected, net_glob)
        cos_sim_matrix, l2_dist_matrix = calculate_model_distances(
            w_locals_selected, net_glob
        )

        # get model norm
        for i in range(args.num_users):
            model_norm = torch.norm(
                flatten_model(w_locals_selected[i])
                - flatten_model(net_glob.state_dict())
            )
            model_norms[i].append(model_norm.cpu().detach().numpy())

        cos_sim_matrix_list.append(cos_sim_matrix)
        l2_dist_matrix_list.append(l2_dist_matrix)

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

    import seaborn as sns

    # create distance folder
    path = ""
    if args.clipping_factor:
        path = f"distances_{args.clipping_factor}"
    elif args.cosine_sim:
        path = f"distances_{args.cosine_lambda}"
    else:
        path = "distances"

    Path(path).mkdir(parents=True, exist_ok=True)
    np.save(
        f"{path}/{args.dataset}_cos_sim_matrix.npy",
        [tensor.cpu().numpy() for tensor in cos_sim_matrix_list],
    )
    np.save(
        f"{path}/{args.dataset}_l2_dist_matrix.npy",
        [tensor.cpu().numpy() for tensor in l2_dist_matrix_list],
    )

    np.save(
        f"{path}/{args.dataset}_model_norms.npy",
        [tensor for tensor in model_norms],
    )

    avg_cos_per_round = torch.stack(cos_sim_matrix_list).mean(axis=0)
    avg_l2_per_round = torch.stack(l2_dist_matrix_list).mean(axis=0)

    # plot the distance avg per epoch
    plt.figure()
    plt.plot(avg_cos_per_round, label="Cosine Similarity")
    plt.plot(avg_l2_per_round, label="L2 Distance")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.legend()
    plt.savefig(f"{path}/{args.dataset}_distance_avg.pdf")
    plt.clf()

    # Customize tick labels
    ax = sns.heatmap(
        cos_sim_matrix,
        annot=True,
        cmap="Blues",
        xticklabels=idxs_users,
        yticklabels=idxs_users,
        fmt=".1f",
    )

    # color_ticklabels(ax, attackers_idx)

    plt.savefig(f"{path}/{args.dataset}_cos_sim_matrix.pdf")
    plt.clf()

    ax = sns.heatmap(
        l2_dist_matrix,
        annot=True,
        cmap="Blues",
        xticklabels=idxs_users,
        yticklabels=idxs_users,
        fmt=".1f",
    )

    # color_ticklabels(ax, attackers_idx)
    plt.savefig(f"{path}/{args.dataset}_l2_dist_matrix.pdf")
    plt.clf()

    # plot the model norms, x axis the number of epochs and y the norm. mark in red bars the attackers
    exit()

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
