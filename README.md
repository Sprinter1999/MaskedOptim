# Masked Optimization

Implementation for our manuscript Robust Federated Learning with Masked Optimization against Distributed Noisy Clients, which is currently under review.
The arxiv version will be released soon.

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

