# Time-Bandits

Code for the paper "Time-Distributed Backdoor Attacks on Federated Spiking Learning".

## Abstract

This paper investigates the vulnerability of spiking neural networks (SNNs) and federated learning (FL) to backdoor attacks using neuromorphic data. Despite the efficiency of SNNs and the privacy advantages of FL, particularly in low-powered devices, we demonstrate that these systems are susceptible to such attacks. We first assess the viability of using FL with SNNs using neuromorphic data, showing its potential usage. Then, we evaluate the transferability of known FL attack methods to SNNs, finding that these lead to suboptimal attack performance. Therefore, we explore backdoor attacks involving single and multiple attackers to improve the attack performance. Our primary contribution is developing a novel attack strategy tailored to SNNs and FL, which distributes the backdoor trigger temporally and across malicious devices, enhancing the attack's effectiveness and stealthiness. In the best case, we achieve a 100% attack success rate, 0.13 MSE, and 98.9 SSIM. Moreover, we adapt and evaluate an existing defense against backdoor attacks, revealing its inadequacy in protecting SNNs. This study underscores the need for robust security measures in deploying SNNs and FL, particularly in the context of backdoor attacks.

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
