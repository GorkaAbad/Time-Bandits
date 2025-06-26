# Time-Bandits

Code for the paper "Time-Distributed Backdoor Attacks on Federated Spiking Learning". Published at ESORICS 2025.

## Abstract

This paper investigates the vulnerability of federated learning (FL) with spiking neural networks (SNNs) to backdoor attacks using neuromorphic data. Despite the efficiency of SNNs and the privacy advantages of FL, particularly in low-powered devices, we demonstrate that these systems are susceptible to such attacks. We first assess the viability of using FL with SNNs using neuromorphic data, showing its potential usage. Then, we evaluate the transferability of known FL attack methods to SNNs, finding that these lead to sub-optimal attack performance. Consequently, we explore backdoor attacks involving single and multiple attackers to improve the attack performance. Our main contribution is developing a novel attack strategy tailored to SNNs and FL, which distributes the backdoor trigger temporally and across malicious clients, enhancing the attack's effectiveness and stealthiness. In the best case, we achieve a 100% attack success rate, 0.13 MSE, and 98.9 SSIM. Moreover, we adapt and evaluate existing defenses against backdoor attacks, revealing their inadequacy in protecting SNNs.

## How to run

### Dependencies

Our code is based on https://github.com/Intelligent-Computing-Lab-Yale/FedSNN. 
First, we have to install those dependencies.

However, we use another framework for training SNNs. https://github.com/fangwei123456/spikingjelly/tree/master.

Consider installing the dependencies contained in the requirements using pip:

```bash
  pip install -r requirements.txt
```

### Preparing the datasets

Some datasets are automatically downloaded. But some others have to be downloaded manually. This is a [restriction](https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/neuromorphic_datasets.html) from the SpikingJelly repo.

Create a `data/` folder in the root of the project:
```bash
mkdir data
```
Then, the datasets should be placed in the following format:
`data/<dataset>/download/`

Check the following section for more details about specific datasets.

#### N-MNIST

N-MNIST **cannot** be downloaded automatically. You have to download it manually from [here](https://www.garrickorchard.com/datasets/n-mnist).
Then, create a folder with the name `mnist` in the `data` folder and put the downloaded files in it.
```bash	
mkdir data/mnist
```

And put the dataset (`.zip` file) in it.
SpikingJelly will automatically unzip the files (creating a `extracted/` folder), and do the rest of the work.

#### CIFAR10-DVS

CIFAR10-DVS **can** be downloaded automatically. You just have to create a folder with the name `cifar10` in the `data` folder.
```bash
mkdir data/cifar10
```

#### DVS128 Gesture

DVS128 Gesture **cannot** be downloaded automatically. You have to download it from [here](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794).
Then, create a folder with the name `gesture` in the `data` folder and put the downloaded files in it.

```bash
mkdir data/gesture
```

And put the dataset (`.gz` file) in it.

### Hardware requirements

In order to run the code a GPU is strongly recommended. 
The code is tested on a machine with 1 NVIDIA A100 GPUs with 20GB.

### Reading the results

After execution, the results are saved in a `.csv` file in the `results/` folder.
The `.csv` file will contain all the execution parameters and the results of the attack, i.e., clean accuracy and backdoor accuracy.


### Running the code

You can run experiments using [`main.py`](main.py). Below are example commands and usage patterns.

#### Basic usage

```bash
python main.py --dataset mnist --num_classes 10 --epochs 10 --local_ep 2 --num_users 10 --frac 0.5 --attacker_idx 0 --split 2 --cupy --warmup
```

### main.py usage options

The script `main.py` supports a variety of command-line arguments to control both the federated learning (FL) setup and the backdoor attack configuration. Below is a summary of the most important options, divided into two categories: **Federated Learning options** and **Attack options**.

#### Federated Learning (FL) options

- `--epochs`: Number of global FL rounds (epochs).  
- `--num_users`: Number of clients/devices in the FL setup.  
- `--frac`: Fraction of clients selected per round (e.g., `0.5` means 50% of clients are selected each round).  
- `--local_ep`: Number of local epochs per client per round.  
- `--local_bs`: Local batch size for each client.  
- `--bs`: Batch size for testing.  
- `--lr`: Learning rate for local training.  
- `--lr_interval`: Intervals (as fractions of total epochs) at which to reduce the learning rate (default: `"0.33 0.66"`).  
- `--alpha`: Degree of non-IID-ness (for data partitioning).  
- `--lr_reduce`: Reduction factor for learning rate.  
- `--timesteps`: Number of timesteps for SNN simulation.  
- `--activation`: SNN activation function (`Linear` or `STDB`).  
- `--optimizer`: Optimizer for SNN backpropagation (`SGD` or `Adam`).  
- `--weight_decay`: Weight decay parameter for the optimizer.  
- `--dropout`: Dropout percentage for conv layers.  
- `--momentum`: SGD momentum.  
- `--dataset`: Dataset to use (`mnist`, `cifar10`, `gesture`, `caltech`).  
- `--data_path`: Path to the dataset folder.  
- `--iid`: Use IID data partitioning (if not set, non-IID is used).  
- `--num_classes`: Number of output classes for the dataset.  
- `--stopping_rounds`: Rounds of early stopping.  
- `--verbose`: Verbose print.  
- `--seed`: Random seed for reproducibility.  
- `--eval_every`: Frequency (in epochs) to evaluate and print metrics.  
- `--result_dir`: Directory to store results.  
- `--train_acc_batches`: Print training progress after this many batches.  
- `--straggler_prob`: Probability of a client being a straggler.  
- `--grad_noise_stdev`: Noise level for gradients.  
- `--cupy`: Use CuPy backend for faster SNN simulation (GPU required).  
- `--patience`: Patience for early stopping.  
- `--warmup`: Pretrain the model before federated learning (uses 10% of total rounds).  

#### Attack options

- `--epsilon`: Poisoning rate (fraction of data to poison for the attacker).  
- `--trigger_size`: Size of the backdoor trigger (e.g., `0.05` for 5% of the image).  
- `--polarity`: Polarity of the trigger.  
- `--pos`: Position of the trigger in the image (e.g., `top-left`).  
- `--target_label`: Target label for the backdoor attack.  
- `--attacker_idx`: Index of the attacker client (set to `0` or another client index to enable attack; if not set, no attack is performed).  
- `--split`: Number of splits for the attack (for Time Bandits attack; splits the trigger among multiple clients).  
- `--scale`: If set, attacker scales their model update to amplify the attack.  
- `--clipping_factor`: Clipping factor for the model norm.  
- `--cosine_sim`: Use cosine similarity for the model norm for the attacker.  
- `--cosine_lambda`: Lambda for the cosine similarity loss.  

#### Example

```bash
python main.py --dataset mnist --num_classes 10 --epochs 10 --local_ep 2 --num_users 25 --frac 0.5 --attacker_idx 0 --epsilon 0.2 --trigger_size 0.05 --scale --split 4 --iid --cupy --warmup
```

For a full list of options and their defaults, run:

```bash
python main.py --help
```