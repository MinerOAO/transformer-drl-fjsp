# Transformer-drl-fjsp
Implementation of Transformer-based deep reinforcement learning method for flexible job shop scheduling problems.
Based on [Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9826438)

## Get Started

### Installation

* Python 3.11.9
* Pytorch 2.5.1
* Gymnasium 0.29.1
* Numpy 1.26.4
* Pandas 2.2.2
* Visdom 0.2.4

### Introduction

* ```data_dev``` and ```data_test``` are the validation sets and test sets, respectively.
* ```env``` The DRL environment
* ```transformer``` Code related to the transformer
* ```model``` Models for testing
* ```save``` Experimental results
* ```utils``` Some helper functions
* ```config.json``` Configuration file
* ```PPO_model.py``` PPO algorithms
* ```test.py``` Test
* ```train.py``` Train
* ```validate.py``` Validation without manual calls
* ```config.json``` Configuration about network model, testing and training.

### train

```
python -m visdom.server
python train.py
```
### test

```
python test.py
```
Note that there should be model files (```*.pt```) in ```./model```.

## Reference

* https://github.com/songwenas12/fjsp-drl

