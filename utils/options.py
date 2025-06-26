import argparse


def args_parser():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument("--epochs", type=int, default=10, help="rounds of training")
    parser.add_argument("--num_users", type=int, default=10, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=1, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=16, help="local batch size: B")
    parser.add_argument("--bs", type=int, default=16, help="test batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr_interval",
        default="0.33 0.66",
        type=str,
        help="intervals at which to reduce lr, expressed as %%age of total epochs",
    )
    parser.add_argument(
        "--alpha", default=0.5, type=float, help="the degree of non-IID-ness"
    )

    parser.add_argument(
        "--lr_reduce", default=10, type=int, help="reduction factor for learning rate"
    )
    parser.add_argument(
        "--timesteps", default=16, type=int, help="simulation timesteps"
    )
    parser.add_argument(
        "--activation",
        default="Linear",
        type=str,
        help="SNN activation function",
        choices=["Linear", "STDB"],
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        help="optimizer for SNN backpropagation",
        choices=["SGD", "Adam"],
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
        help="weight decay parameter for the optimizer",
    )
    parser.add_argument(
        "--dropout", default=0.3, type=float, help="dropout percentage for conv layers"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum (default: 0.5)"
    )

    # Attack arguments
    parser.add_argument("--epsilon", type=float, default=0.1, help="poisoning rate")
    parser.add_argument(
        "--trigger_size", type=float, default=0.3, help="size of the trigger"
    )
    parser.add_argument(
        "--polarity", type=int, default=3, help="polarity of the trigger"
    )
    parser.add_argument(
        "--pos", type=str, default="top-left", help="position of the trigger"
    )
    parser.add_argument("--target_label", type=int, default=0, help="target label")
    parser.add_argument("--attacker_idx", type=int, default=None, help="attacker index")
    parser.add_argument(
        "--split", type=int, default=None, help="number of splits for the attack"
    )
    parser.add_argument("--scale", action="store_true", help="use the scaling attack")

    # other arguments
    parser.add_argument(
        "--dataset", type=str, default="gesture", help="name of dataset"
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="directory of dataset"
    )
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument(
        "--stopping_rounds", type=int, default=10, help="rounds of early stopping"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose print")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Frequency of model evaluation"
    )
    parser.add_argument(
        "--result_dir", type=str, default="results", help="Directory to store results"
    )
    parser.add_argument(
        "--train_acc_batches",
        default=200,
        type=int,
        help="print training progress after this many batches",
    )
    parser.add_argument(
        "--straggler_prob", type=float, default=0.0, help="straggler probability"
    )
    parser.add_argument(
        "--grad_noise_stdev", type=float, default=0.0, help="Noise level for gradients"
    )
    parser.add_argument(
        "--cupy", action="store_true", help="Whether to use cupy for faster computation"
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="Patience for early stopping"
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Whether to use warmup training for FL as 10% of the total rounds",
    )
    parser.add_argument(
        "--clipping_factor",
        type=float,
        default=None,
        help="Clipping factor for the model norm",
    )
    parser.add_argument(
        "--cosine_sim",
        default=False,
        action="store_true",
        help="Whether to use cosine similarity for the model norm for the attacker",
    )
    parser.add_argument(
        "--cosine_lambda",
        type=float,
        default=0.1,
        help="Lambda for the cosine similarity loss",
    )
    args = parser.parse_args()
    return args
