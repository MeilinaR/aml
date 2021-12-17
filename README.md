# Automated Machine Learning Project
## Learning to Learn by Gradient Descent by Gradient Descent
Much of the code in this repository stems from https://github.com/deepmind/learning-to-learn. The remainder of this README, below the two horizontal lines is copied from that repo as well. While the repository can be run locally, we strongly recommend running the code in Google Colab due to some breaking changes in versions of `tensorflow` and `sonnet` which we have taken care of in our own notebook. The notebook can be found [https://colab.research.google.com/drive/1NdXeetzPzrpyXSlNYpOlBGlbTY8oh4AJ?usp=sharing#scrollTo=tBEtC-oOBX-F](here). Since training the L2L optimizer can take some time, we have provided the network weights from our own experiments in the `saved_optimizers` folder.

The original repo supported two optimizers: `Adam`, and the `L2L` optimizer used in the original 'learning to learn by gradient descent by gradient descent' paper. We have added two more: the `fixed` lambda optimizer, and the `learned` lambda optimizer. In the training phase, the optimizer can be specified using the `--which_aml={OPTIM}` flag, where `{OPTIM}` can be `fixed` or `learned` - omitting the flag defaults to `L2L`. Adam does not require a training phase. The network  In the evaluation phase, the optimizer is specified by the `--optimizer={OPTIM}` flag, where `{OPTIM}` is `Adam`, `L2L`, `fixed` or `learned`. Omitting this flag defaults to `L2L` as well. 

---

---


# [Learning to Learn](https://arxiv.org/abs/1606.04474) in TensorFlow


## Dependencies

* [TensorFlow >=1.0](https://www.tensorflow.org/)
* [Sonnet >=1.0](https://github.com/deepmind/sonnet)


## Training

```
python train.py --problem=mnist --save_path=./mnist
```

Command-line flags:

* `save_path`: If present, the optimizer will be saved to the specified path
    every time the evaluation performance is improved.
* `num_epochs`: Number of training epochs.
* `log_period`: Epochs before mean performance and time is reported.
* `evaluation_period`: Epochs before the optimizer is evaluated.
* `evaluation_epochs`: Number of evaluation epochs.
* `problem`: Problem to train on. See [Problems](#problems) section below.
* `num_steps`: Number of optimization steps.
* `unroll_length`: Number of unroll steps for the optimizer.
* `learning_rate`: Learning rate.
* `second_derivatives`: If `true`, the optimizer will try to compute second
    derivatives through the loss function specified by the problem.


## Evaluation

```
python evaluate.py --problem=mnist --optimizer=L2L --path=./mnist
```

Command-line flags:

* `optimizer`: `Adam` or `L2L`.
* `path`: Path to saved optimizer, only relevant if using the `L2L` optimizer.
* `learning_rate`: Learning rate, only relevant if using `Adam` optimizer.
* `num_epochs`: Number of evaluation epochs.
* `seed`: Seed for random number generation.
* `problem`: Problem to evaluate on. See [Problems](#problems) section below.
* `num_steps`: Number of optimization steps.


## Problems

The training and evaluation scripts support the following problems (see
`util.py` for more details):

* `simple`: One-variable quadratic function.
* `simple-multi`: Two-variable quadratic function, where one of the variables
    is optimized using a learned optimizer and the other one using Adam.
* `quadratic`: Batched ten-variable quadratic function.
* `mnist`: Mnist classification using a two-layer fully connected network.
* `cifar`: Cifar10 classification using a convolutional neural network.
* `cifar-multi`: Cifar10 classification using a convolutional neural network,
    where two independent learned optimizers are used. One to optimize
    parameters from convolutional layers and the other one for parameters from
    fully connected layers.


New problems can be implemented very easily. You can see in `train.py` that
the `meta_minimize` method from the `MetaOptimizer` class is given a function
that returns the TensorFlow operation that generates the loss function we want
to minimize (see `problems.py` for an example).

It's important that all operations with Python side effects (e.g. queue
creation) must be done outside of the function passed to `meta_minimize`. The
`cifar10` function in `problems.py` is a good example of a loss function that
uses TensorFlow queues.


Disclaimer: This is not an official Google product.
