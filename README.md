# Masked Optimization

Implementation for our manuscript `Robust Federated Learning against Noisy Clients via Masked Optimization`, which is currently under review for journal.
The arxiv version is available via [this link](https://arxiv.org/abs/2506.02079).

## Supported Methods in this code base
- FedAvg
- TrimmedMean
- Median
- Krum
- maskedOptim (the proposed method)

To evaluate these methods, you can modify the method configuration in  `eval.sh`.

## Datasets
CIFAR-10, CIFAR-N-10, AGNews
For Clothing1M, please refer to our another open-sourced  [code base](https://github.com/Sprinter1999/Clothing1M_FedAvg) for further experiments.

